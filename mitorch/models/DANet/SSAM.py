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
from torch.nn import Parameter


class SpatialAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = self_attention_attr.MIDDLE_REDUCTION_RATIO
        self.modulation_type = self_attention_attr.INTERNAL_MODULATION_TYPE

        self._create_net(self_attention_attr)

    def _create_net(self, self_attention_attr):
        in_channels, out_channels = self.gate_channels, self.gate_channels // self.reduction_ratio

        self.query_conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1))

        self.key_conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1))

        self.value_conv_layer = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1, 1))

        self.gamma = Parameter(torch.zeros(1))

        self.attention_normalization = nn.Softmax(dim=-1)

        self.modulation_layer = ModulationAggregationBlock(
            gate_channels=self.gate_channels,
            modulation_type=self.modulation_type
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_query = self.query_conv_layer(x)
        x_query = x_query.view(B, C, -1).permute(0, 2, 1)

        x_key = self.key_conv_layer(x)
        x_key = x_key.view(B, C, -1)

        x_energy = torch.bmm(x_query, x_key)

        x_attention = self.attention_normalization(x_energy)

        x_attention = x_attention.permute(0, 2, 1)  # why? paper does not explain!

        x_value = self.value_conv_layer(x)
        x_value = x_value.view(B, C, -1)

        x_out = torch.bmm(x_value, x_attention)

        x_out = x_out.view(B, C, D, H, W)

        x_out = self.gamma * x_out

        x_out = self.modulation_layer((x, x_out))

        return x_out
