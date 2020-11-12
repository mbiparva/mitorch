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


class SpatialAttentionModule(nn.Module):
    def __init__(self, gate_channels, self_attention_attr):
        super().__init__()
        self.gate_channels = gate_channels
        self.kernel_size = tuple([self_attention_attr.KERNEL_SIZE] * 3)
        self.input_reduction_ratio = self_attention_attr.INPUT_REDUCTION_RATIO
        self.middle_reduction_ratio = self_attention_attr.MIDDLE_REDUCTION_RATIO
        self.modulation_type = self_attention_attr.INTERNAL_MODULATION_TYPE

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
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=1,
                kernel_size=(1, 1, 1),
                bias=False,
            )
        )

    def forward(self, x):
        x_out = self.conv_layers(x)

        x_out = x_out.expand_as(x)

        return x_out