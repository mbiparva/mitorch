#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import MODEL_REGISTRY
from models.weight_init_helper import init_weights
from utils.models import pad_if_necessary
from utils.models import pad_if_necessary_all
from models.Unet3D import Encoder as Unet3DEncoder, Decoder as Unet3DDecoder, SegHead as Unet3DSegHead
from models.Unet3D import Unet3D, BasicBlock, ContextBlock, is_3d
from models.NestedUnet3D import ModulationBlock


IS_3D = True


class MLPModule(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU,
            nn.Linear(out_channels, in_channels),
        )


class ChannelAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.pooling_type = self_attention_attr.CHANNEL_POOLING_TYPE
        assert self.pooling_type in ('max', 'average', 'pa', 'lse'), 'p is undefined'
        self.reduction_ratio = self_attention_attr.REDUCTION_RATIO

        self._create_net()

    def _create_net(self):
        self.mlp_layer = MLPModule(
            in_channels=self.gate_channels,
            out_channels=self.gate_channels // self.reduction_ratio
        )

    def forward_pooling(self, pooling_type, x):
        B, C, D, H, W = x.shape

        if pooling_type == 'max':
            x = F.max_pool3d(x, kernel_size=(D, H, W))
        elif pooling_type == 'average':
            x = F.avg_pool3d(x, kernel_size=(D, H, W))
        elif pooling_type == 'pa':
            x = x.view(B * C, -1)
            x = F.lp_pool1d(x, 2, kernel_size=(D * H * W))
            pass
        elif pooling_type == 'lse':
            x = x.view(B, C, -1)
            x = torch.logsumexp(x, dim=-1, keepdim=True)
            pass
        else:
            raise NotImplementedError()

        x = x.view(B, -1)

        x = self.mlp_layer(x)

        return x

    def forward(self, x):
        x_out = self.forward_pooling(self.pooling_type, x)

        x_out = x_out.expand_as(x)

        return x_out


class SpatialAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.kernel_size = tuple([self_attention_attr.SPATIAL_KERNEL_SIZE] * 3)
        self.num_conv_blocks = self_attention_attr.NUM_CONV_BLOCKS
        self.dilation = tuple([self_attention_attr.DILATION] * 3)
        self.reduction_ratio = self_attention_attr.REDUCTION_RATIO

        self._create_net(self_attention_attr)

    @staticmethod
    def get_layer_name(i, postfix=''):
        return 'bam_sam_layer{:03}{}'.format(i, postfix)

    def _create_net(self, self_attention_attr):
        in_channels, out_channels = self.gate_channels, self.gate_channels // self.reduction_ratio
        self.conv_layers = nn.Sequential()
        self.conv_layers.add_module(
            'base',
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
            )
        )

        for i in range(self.num_conv_blocks):
            self.add_module(
                self.get_layer_name(i),
                BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                )
            )

        self.conv_layers.add_module(
            'last',
            nn.Conv3d(in_channels=out_channels, out_channels=1, kernel_size=(1, 1, 1), bias=False)
        )

    def forward(self, x):
        x_out = self.conv_layers(x)

        x_out = x_out.expand_as(x)

        return x_out


class ModulationAggregationBlock(nn.Module):
    def __init__(self, gate_channels, modulation_type, concat_reduction_factor=2, conv_only=True):
        super().__init__()
        self.gate_channels = gate_channels
        self.modulation_type = modulation_type
        self.concat_reduction_factor = concat_reduction_factor
        self.conv_only = conv_only

        self._create_net()

    def _create_conv(self, in_channels, out_channels):
        if self.conv_only:
            return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False)
        else:
            return BasicBlock(in_channels, out_channels, kernel_size=(1, 1, 1))

    def _create_concat_squeeze(self):
        in_channels, out_channels = self.concat_reduction_factor * self.gate_channels, self.gate_channels

        return self._create_conv(in_channels, out_channels)

    def _create_net(self):
        self.attention_modulation = ModulationBlock(self.modulation_type)
        if self.modulation_type == 'concatenation':
            self.concat_reduc_conv_layer = self._create_concat_squeeze()

    def forward(self, x):
        x = self.attention_modulation(x)
        if self.modulation_type == 'concatenation':
            x = self.concat_reduc_conv_layer(x)

        return x


class GAMBlock(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.channel, self.spatial = self_attention_attr.CHANNEL, self_attention_attr.SPATIAL
        self.cm_modulation_type = self_attention_attr.CROSS_MODAL_MODULATION_TYPE
        self.ref_modulation_type = self_attention_attr.REF_MODULATION_TYPE
        self.residual = self_attention_attr.RESIDUAL
        assert self.channel or self.spatial, 'either modalities must be on'

        self._create_net(self_attention_attr)

    def _create_net(self, self_attention_attr):
        if self.channel:
            self.channel_attention_layer = ChannelAttentionModule(self.gate_channels, self_attention_attr)
        if self.spatial:
            self.spatial_attention_layer = SpatialAttentionModule(self.gate_channels, self_attention_attr)

        if self.channel is self.spatial is True:
            self.cm_att_mod_agg = ModulationAggregationBlock(self.gate_channels, self.cm_modulation_type)

        self.attention_normalization = nn.Sigmoid()

        self.ref_att_mod_agg = ModulationAggregationBlock(self.gate_channels, self.ref_modulation_type)

    def forward(self, x):
        x_channel_attention_map, x_spatial_attention_map, x_attention_map = None, None, None

        if self.channel:
            x_channel_attention_map = x_attention_map = self.channel_attention_layer(x)
        if self.spatial:
            x_spatial_attention_map = x_attention_map = self.spatial_attention_layer(x)

        if self.channel is self.spatial is True:
            x_attention_map = self.cm_att_mod_agg((x_channel_attention_map, x_spatial_attention_map))

        x_attention_map = self.attention_normalization(x_attention_map)

        x_attention_map = self.ref_att_mod_agg((x, x_attention_map))

        x = (x + x_attention_map) if self.residual else x

        return x
