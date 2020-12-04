#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import (
            senet154,
            se_resnet50,
            se_resnet101,
            se_resnet152,
            se_resnext50_32x4d,
            se_resnext101_32x4d,
        )
from utils.MONAI_networks.factories import Act, Norm

IS_3D = True


@MODEL_REGISTRY.register()
class SEUnet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        # TODO add them to net cfg settings for hpo
        _net_sel = 0
        _net_caller = (
            senet154,
            se_resnet50,
            se_resnet101,
            se_resnet152,
            se_resnext50_32x4d,
            se_resnext101_32x4d,
        )[_net_sel]
        self.EncoDecoSeg = _net_caller(
            spatial_dims=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            pretrained=False,
            progress=True,
        )

    def forward_core(self, x):
        x = self.EncoDecoSeg(x)
        return x
