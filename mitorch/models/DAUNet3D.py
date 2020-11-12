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
from models.Unet3D import Decoder as Unet3DDecoder, SegHead as Unet3DSegHead
from models.Unet3D import Unet3D, CompoundBlock, is_3d
from models.NestedUnet3D import ModulationBlock
from models.Unet3DCBAM import Unet3DCBAM, Encoder as Unet3DCBAMEncoder
from models.DANet import LAMBlock
import models.DANet

IS_3D = True


class Encoder(Unet3DCBAMEncoder):
    def __init__(self, cfg, local_sab):
        super().__init__(cfg, local_sab)


class Decoder(Unet3DDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)


class SegHead(Unet3DSegHead):
    def __init__(self, cfg):
        super().__init__(cfg)


@MODEL_REGISTRY.register()
class DAUnet3D(Unet3DCBAM):
    """
    This is a Unet3D variant with self-attention modules. It is extended to
    work in 3D segmentation networks. The SA modules are based on batch matrix matrix (bmm) multiply
    and are inspired by the CVPR'19 paper:
        (1) Dual attention network for scene segmentation"
    The implementation is motivated from the original PyTorch code base at:
        https://github.com/junfu1115/DANet/blob/master/encoding/models/sseg/danet.py
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'
        models.DANet.LAModule.IS_3D = IS_3D
        super().set_processing_mode()

    def _create_net(self):
        self.Encoder = Encoder(self.cfg, local_sab=LAMBlock)
        self.Decoder = Decoder(self.cfg)
        self.SegHead = SegHead(self.cfg)
