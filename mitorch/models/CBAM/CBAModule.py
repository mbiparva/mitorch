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
            nn.ReLU,
            nn.Linear(out_channels, in_channels),
        )


class ChannelAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.pooling_types = self_attention_attr.CBAM_CHANNEL_POOLING_TYPES
        assert isinstance(self.pooling_types, (tuple, list)) and len(self.pooling_types) > 0
        assert all([True if p in ('max', 'average', 'pa', 'lse') else False for p in self.pooling_type]), 'p undefined'
        self.reduction_ratio = self_attention_attr.REDUCTION_RATIO

        self._create_net()

    def _create_net(self):
        self.mlp_layer = MLPModule(
            in_channels=self.gate_channels,
            out_channels=self.gate_channels // self.reduction_ratio
        )
        self.modulation_layer = ModulationBlock(modulation_type='sum')  # TODO could type other types
        self.attention_normalization = nn.Sigmoid()
        self.final_modulation_layer = ModulationBlock(modulation_type='multiplicative')

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
        B, C, D, H, W = x.shape

        attention_maps = list()
        for p in self.pooling_types:
            x_out = self.forward_pooling(p, x)
            attention_maps.append(x_out)

        x_out = self.modulation_layer(attention_maps) if len(attention_maps) > 1 else attention_maps[0]

        x_out = self.attention_normalization(x_out)
        x_out = x_out.view(B, C, 1, 1, 1).expand_as(x)

        x_out = self.final_modulation_layer((x, x_out))

        return x_out


class ChannelPoolingLayer(nn.Module):
    def __init__(self, pooling_type):
        super().__init__()
        assert pooling_type in ('max', 'average'), 'pooling type is undefined'

        self._create_net(pooling_type)

    def _create_net(self, pooling_type):
        if pooling_type == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool2d((1, None))
        else:
            self.pooling_layer = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1)
        x = self.pooling_layer(x)
        x = x.view(B, C, D, H, W)

        return x


class ChannelPoolingBlock(nn.Module):
    def __init__(self, self_attention_attr):
        super().__init__()
        self.pooling_type = self_attention_attr.SPATIAL_POOLING_TYPE
        assert self.pooling_type in ('max', 'average', 'max_average'), 'pooling type is undefined'

        self._create_net()

    def _create_net(self):
        self.max_pooling = ChannelPoolingLayer('max')
        self.avg_pooling = ChannelPoolingLayer('average')
        self.modulation_layer = ModulationBlock(modulation_type='concatenation')  # TODO could type other types
        if self.pooling_type == 'max' or self.pooling_type == 'average':
            self.out_channels = 1
        else:
            self.out_channels = 2

    def forward(self, x):
        if self.pooling_type == 'max':
            return self.max_pooling(x)
        elif self.pooling_type == 'average':
            return self.avg_pooling(x)
        else:
            x_max = self.max_pooling(x)
            x_avg = self.avg_pooling(x)

            return self.modulation_layer((x_max, x_avg))


class SpatialAttentionModule(nn.Module):
    def __init__(self, self_attention_attr):
        super().__init__()
        self.kernel_size = tuple([self_attention_attr.SPATIAL_KERNEL_SIZE] * 3)

        self._create_net(self_attention_attr)

    def _create_net(self, self_attention_attr):
        self.channel_pooling_layer = ChannelPoolingBlock(self_attention_attr)
        in_channels, out_channels = self.channel_pooling_layer.out_channels, 1
        CONV_ONLY = (False, True)[0]
        if CONV_ONLY:
            self.spatial_conv_layer = nn.Conv3d(in_channels, out_channels,
                                                kernel_size=self.kernel_size,
                                                padding=tuple((torch.tensor(self.kernel_size) // 2).tolist()),
                                                bias=False)
        else:
            self.spatial_conv_layer = BasicBlock(in_channels, out_channels, kernel_size=self.kernel_size)
        self.attention_normalization = nn.Sigmoid()
        self.modulation_layer = ModulationBlock(modulation_type='multiplicative')

    def forward(self, x):
        x_attention_map = self.channel_pooling_layer(x)

        x_attention_map = self.spatial_conv_layer(x_attention_map)

        x_attention_map = self.attention_normalization(x_attention_map)

        x = self.modulation_layer((x, x_attention_map))

        return x


class CBAMBlock(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.channel, self.spatial = self_attention_attr.CHANNEL, self_attention_attr.SPATIAL
        self.modulation_type = self_attention_attr.CBAM_REF_MODULATION_TYPE
        assert self.channel or self.spatial, 'either modalities must be on'

        self._create_net(self_attention_attr)

    def _create_concat_squeeze(self):
        if self.channel or self.spatial:
            concat_reduction_factor = 2
        elif self.channel and self.spatial:
            concat_reduction_factor = 3
        else:
            concat_reduction_factor = 3
        assert concat_reduction_factor

        in_channels, out_channels = concat_reduction_factor * self.gate_channels, self.gate_channels

        CONV_ONLY = (False, True)[0]
        if CONV_ONLY:
            self.concat_reduc_conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False)
        else:
            self.concat_reduc_conv_layer = BasicBlock(in_channels, out_channels, kernel_size=(1, 1, 1))

    def _create_net(self, self_attention_attr):
        if self.channel:
            self.channel_attention_layer = ChannelAttentionModule(self.gate_channels, self_attention_attr)
            self.channel_attention_modulation = ModulationBlock(self.modulation_type)
        if self.spatial:
            self.spatial_attention_layer = SpatialAttentionModule(self_attention_attr)
            self.spatial_attention_modulation = ModulationBlock(self.modulation_type)

        self.type_concat = self.modulation_type == 'concatenation'
        if self.type_concat:
            self._create_concat_squeeze()

    def forward(self, x):
        if self.channel:
            x_channel_attention_map = self.channel_attention_layer(x)
            x = self.channel_attention_modulation((x, x_channel_attention_map))
        if self.spatial:
            x_spatial_attention_map = self.spatial_attention_layer(x)
            x = self.spatial_attention_modulation((x, x_spatial_attention_map))

        if self.type_concat:
            x = self.concat_reduc_conv_layer(x)

        return x