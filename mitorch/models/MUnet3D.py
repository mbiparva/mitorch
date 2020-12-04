#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import UNet
from utils.MONAI_networks.factories import Act, Norm

IS_3D = True


@MODEL_REGISTRY.register()
class MUnet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        # TODO add them to net cfg settings for hpo
        self.EncoDecoSeg = UNet(
            dimensions=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            out_channels=self.cfg.MODEL.NUM_CLASSES,
            channels=(32, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=0,
            act=Act.PRELU,
            norm=Norm.INSTANCE,
            dropout=self.cfg.MODEL.DROPOUT_RATE,
        )

    def forward_core(self, x):
        x = self.EncoDecoSeg(x)
        return x
