#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch.nn as nn
from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import (
            densenet121,
            densenet169,
            densenet201,
            densenet264,
)
from .Unet3D import Decoder, SegHead


IS_3D = True


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._create_net()

    def _create_net(self):
        # TODO add them to net cfg settings for hpo
        _net_sel = 0
        _net_caller = (
            densenet121,
            densenet169,
            densenet201,
            densenet264,
        )[_net_sel]
        self.net = _net_caller(
            pretrained=False,
            progress=True,
            **{
                'spatial_dims': 3,
                'in_channels': self.cfg.MODEL.INPUT_CHANNELS,
                'out_channels': self.cfg.MODEL.NUM_CLASSES,
                'init_features': 64,
                'dropout_prob': self.cfg.MODEL.DROPOUT_RATE,
                'growth_rate': 32,
                'block_config': (6, 12, 24, 16),
                'bn_size': 4,
            }
        )

    def forward(self, x):
        output_list = list()
        for name, module in self.net.features.named_modules():
            x = module(x)
            if name == 'norm5' or name.startswith('transition'):
                output_list.append(x)
        return output_list


@MODEL_REGISTRY.register()
class Densenet3D(NetABC):
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
