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