#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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


class EnergyMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self._create_net()

    def _create_net(self):
        self.max_pool_layer = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x_max = self.max_pool_layer(x)

        x_max = x_max.expand_as(x)

        x = x_max - x

        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.modulation_type = self_attention_attr.INTERNAL_MODULATION_TYPE
        self.ENERGY_MAX_NORM = (False, True)[1]

        self._create_net()

    def _create_net(self):
        if self.ENERGY_MAX_NORM:
            self.energy_maximize_layer = EnergyMaxNorm()

        self.gamma = Parameter(torch.zeros(1))

        self.attention_normalization = nn.Softmax(dim=-1)

        self.modulation_layer = ModulationAggregationBlock(
            gate_channels=self.gate_channels,
            modulation_type=self.modulation_type
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_query = x.view(B, C, -1)
        x_key = x.view(B, C, -1).permute(0, 2, 1)

        x_energy = torch.bmm(x_query, x_key)
        if self.ENERGY_MAX_NORM:
            x_energy = self.energy_maximize_layer(x_energy)

        x_attention = self.attention_normalization(x_energy)

        x_value = x.view(B, C, -1)
        x_out = torch.bmm(x_attention, x_value)

        x_out = x_out.view(B, C, D, H, W)

        x_out = self.gamma * x_out

        x_out = self.modulation_layer((x, x_out))

        return x_out
