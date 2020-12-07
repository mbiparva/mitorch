#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch.nn as nn
from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import HighResNet
from .Unet3D import Decoder, SegHead
from utils.MONAI_networks.highresnet import Normalisation, Activation, DEFAULT_LAYER_PARAMS_3D, MarkerLayer


IS_3D = True


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._create_net()

    def _create_net(self):
        # TODO add them to net cfg settings for hpo
        self.net = HighResNet(
            spatial_dims=3,
            in_channels=self.cfg.MODEL.INPUT_CHANNELS,
            out_channels=self.cfg.MODEL.NUM_CLASSES,
            norm_type=Normalisation.BATCH,
            acti_type=Activation.RELU,
            dropout_prob=self.cfg.MODEL.DROPOUT_RATE,
            layer_params=DEFAULT_LAYER_PARAMS_3D,
        )

    def forward(self, x):
        output_list = list()
        encoder_modules = list(self.net.features.modules())[:-1]  # the last layer is for classification
        for module in encoder_modules:
            x = module(x)
            if isinstance(module, MarkerLayer):
                output_list.append(x)
        return output_list


@MODEL_REGISTRY.register()
class Highresnet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        self.Encoder = Encoder(self.cfg)
        self.Decoder = Decoder(self.cfg)
        self.SegHead = SegHead(self.cfg)

    def forward_core(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)  # TODO check to see the specs of the outputs matches the defaults decoder specs
        x = self.SegHead(x)

        return x
