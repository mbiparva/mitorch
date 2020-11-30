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
from models.CBAM.GAModule import ModulationAggregationBlock
from models.DANet.CSAM import ChannelAttentionModule
from models.DANet.SSAM import SpatialAttentionModule

IS_3D = True


class AttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr, attention_modality):
        super().__init__()
        self.gate_channels = gate_channels
        self.input_reduction_ratio = self_attention_attr.INPUT_REDUCTION_RATIO
        self.kernel_size = tuple([self_attention_attr.KERNEL_SIZE] * 3)

        self._create_net(self_attention_attr, attention_modality)

    def _create_net(self, self_attention_attr, attention_modality):
        in_channels, out_channels = self.gate_channels, self.gate_channels // self.input_reduction_ratio
        assert out_channels > 0, 'reduce reduction ration, out_channels hit < 1 range'

        self.pre_attention_conv_layer = BasicBlock(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=self.kernel_size)
        self.attention_block = attention_modality(out_channels, self_attention_attr)
        self.post_attention_conv_layer = BasicBlock(in_channels=out_channels, out_channels=out_channels,
                                                    kernel_size=self.kernel_size)

        self.last_conv_layer = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.pre_attention_conv_layer(x)

        x = self.attention_block(x)

        x = self.post_attention_conv_layer(x)

        x = self.last_conv_layer(x)

        return x


class LAMBlock(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.channel, self.spatial = self_attention_attr.CHANNEL, self_attention_attr.SPATIAL
        self.cm_modulation_type = self_attention_attr.CROSS_MODAL_MODULATION_TYPE
        self.ref_modulation_type = self_attention_attr.REF_MODULATION_TYPE
        assert self.channel or self.spatial, 'either modalities must be on'

        self._create_net(self_attention_attr)

    def _create_net(self, self_attention_attr):
        if self.channel:
            self.channel_attention_layer = AttentionModule(self.gate_channels, self_attention_attr,
                                                           ChannelAttentionModule)
        if self.spatial:
            self.spatial_attention_layer = AttentionModule(self.gate_channels, self_attention_attr,
                                                           SpatialAttentionModule)

        if self.channel is self.spatial is True:
            self.cm_att_mod_agg = ModulationAggregationBlock(self.gate_channels, self.cm_modulation_type)

        self.attention_final_conv_layer = nn.Conv3d(in_channels=self.gate_channels, out_channels=self.gate_channels,
                                                    kernel_size=(1, 1, 1), bias=False)

        self.ref_att_mod_agg = ModulationAggregationBlock(self.gate_channels, self.ref_modulation_type)

    def forward(self, x):
        x_channel_attention_map, x_spatial_attention_map, x_attention_map = None, None, None

        if self.channel:
            x_channel_attention_map = x_attention_map = self.channel_attention_layer(x)
        if self.spatial:
            x_spatial_attention_map = x_attention_map = self.spatial_attention_layer(x)

        if self.channel is self.spatial is True:
            x_attention_map = self.cm_att_mod_agg((x_channel_attention_map, x_spatial_attention_map))

        x_attention_map = self.attention_final_conv_layer(x_attention_map)

        x_attention_map = self.ref_att_mod_agg((x, x_attention_map))

        return x_attention_map
