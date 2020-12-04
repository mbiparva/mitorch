#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

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
        # TODO add them to net cfg settings for hpo
        self.EncoDecoSeg = BasicUNet(
            dimensions=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            out_channels=self.cfg.MODEL.NUM_CLASSES,
            features=(32, 32, 64, 128, 256, 32),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            dropout=self.cfg.MODEL.DROPOUT_RATE,
            upsample=("deconv", "nontrainable")[0],
        )

    def forward_core(self, x):
        x = self.EncoDecoSeg(x)
        return x
