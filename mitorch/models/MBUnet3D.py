#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch.nn as nn
from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import BasicUNet

IS_3D = True


@MODEL_REGISTRY.register()
class MBUnet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        self.EncoDecoSeg = BasicUNet(
            dimensions=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            out_channels=self.cfg.MODEL.NUM_CLASSES,
            features=self.cfg.MODEL.SETTINGS.FEATURES,
            act=self.cfg.MODEL.SETTINGS.ACT,
            norm=self.cfg.MODEL.SETTINGS.NORM,
            dropout=self.cfg.MODEL.DROPOUT_RATE,
            upsample=self.cfg.MODEL.SETTINGS.UPSAMPLE,
        )

    def forward_core(self, x):
        x = self.EncoDecoSeg(x)

        return x
