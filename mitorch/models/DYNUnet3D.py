#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import DynUNet
from utils.MONAI_networks.factories import Act, Norm

IS_3D = True


@MODEL_REGISTRY.register()
class DYNUnet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        # TODO add them to net cfg settings for hpo
        self.EncoDecoSeg = DynUNet(
            spatial_dims=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            out_channels=self.cfg.MODEL.NUM_CLASSES,
            kernel_size=(3, 3, 3, 3, 3),
            strides=(1, 2, 2, 2, 2),
            upsample_kernel_size=(2, 2, 2, 2, 2),
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=2,
            res_block=True,
        )

    def forward_core(self, x):
        x = self.EncoDecoSeg(x)
        return x
