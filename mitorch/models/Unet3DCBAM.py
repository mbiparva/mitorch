#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_REGISTRY
from .weight_init_helper import init_weights
from utils.models import pad_if_necessary
from utils.models import pad_if_necessary_all
from models.Unet3D import Encoder as Unet3DEncoder, Decoder as Unet3DDecoder, SegHead as Unet3DSegHead
from models.Unet3D import Unet3D, BasicBlock, ContextBlock, is_3d
from models.NestedUnet3D import ModulationBlock
from models.CBAM import CBAMBlock, BAMBlock
import models.CBAM

IS_3D = True


class CompoundBlock(nn.Module):
    def __init__(self, i, in_channels, out_channels, stride, dilation, p, self_attention, self_attention_attr):
        super().__init__()
        self.self_attention = self_attention
        self.residual_relative = self_attention_attr.RESIDUAL_RELATIVE

        self._create_net(i, in_channels, out_channels, stride, dilation, p, self_attention_attr)

    def _create_net(self, i, in_channels, out_channels, stride, dilation, p, self_attention_attr):
        kwargs = dict()
        if i == 0:
            kwargs['dilation'] = dilation
        else:
            kwargs['stride'] = stride
        self.input_layer = BasicBlock(in_channels, out_channels, **kwargs)
        self.context_layer = ContextBlock(out_channels, out_channels, dilation=dilation, p=p)

        if self.self_attention:
            self.self_attention_layer = CBAMBlock(out_channels, self_attention_attr)

    def forward(self, x):
        x_input = self.input_layer(x)
        x = self.context_layer(x_input)
        if self.self_attention and self.residual_relative == 'before':
            x = self.self_attention_layer(x)
        x = x_input + x
        if self.self_attention and self.residual_relative == 'after':
            x = self.self_attention_layer(x)
        return x


class Encoder(Unet3DEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def get_layer_name(i, postfix=''):
        return 'decoder_layer{:03}{}'.format(i, postfix)

    def _create_net(self):
        in_channels = self.cfg.MODEL.INPUT_CHANNELS
        for i in range(self.cfg.MODEL.ENCO_DEPTH):
            out_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i
            self.add_module(
                self.get_layer_name(i),
                CompoundBlock(i, in_channels, out_channels, stride=self.stride, dilation=self.dilation, p=self.p,
                              self_attention=i in self.cfg.MODEL.SETTINGS.CBAM.BLOCKS,
                              self_attention_attr=self.cfg.MODEL.SETTINGS.CBAM),
            )
            if i in self.cfg.MODEL.SETTINGS.BAM.BLOCKS:
                self.add_module(
                    self.get_layer_name(i, postfix='_BAM'),
                    BAMBlock(out_channels, self_attention_attr=self.cfg.MODEL.SETTINGS.BAM),
                )

            in_channels = out_channels

    def forward(self, x):
        outputs = list()
        for i in range(self.cfg.MODEL.ENCO_DEPTH):
            x = getattr(self, self.get_layer_name(i))(x)
            if i in self.cfg.MODEL.SETTINGS.BAM.BLOCKS:
                x = getattr(self, self.get_layer_name(i, '_BAM'))(x)
            outputs.append(x)
        return outputs


class Decoder(Unet3DDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)


class SegHead(Unet3DSegHead):
    def __init__(self, cfg):
        super().__init__(cfg)


@MODEL_REGISTRY.register()
class Unet3DCBAM(Unet3D):
    """
    This is a Unet3D variant with self-attention modules. It is extended to
    work in 3D segmentation networks. The SA modules are pooling-based
    and are inspired by the two papers:
        (1) BAM: Bottleneck Attention Module (BMVC2018)"
        (2) CBAM: Convolutional Block Attention Module (ECCV2018)"
    The implementation is motivated from the original PyTorch code base at:
        https://github.com/Jongchan/attention-module
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'
        models.CBAM.CBAModule.IS_3D = models.CBAM.BAModule.IS_3D = IS_3D
        super().set_processing_mode()

    def _create_net(self):
        self.Encoder = Encoder(self.cfg)
        self.Decoder = Decoder(self.cfg)
        self.SegHead = SegHead(self.cfg)
