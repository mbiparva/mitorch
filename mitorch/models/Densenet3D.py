#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn
from .build import MODEL_REGISTRY
from models.NetABC import NetABC
from utils.MONAI_networks import (
            densenet121,
            densenet169,
            densenet201,
            densenet264,
)
from .Unet3D import pad_if_necessary, ParamUpSamplingBlock, LocalizationBlock
from utils.MONAI_networks.utils import MarkerLayer
from utils.models import is_3d


IS_3D = True


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._create_net()

    def _create_net(self):
        _net_sel = self.cfg.MODEL.SETTINGS._NET_SEL
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
                'dropout_prob': self.cfg.MODEL.DROPOUT_RATE,
                'bn_size': self.cfg.MODEL.SETTINGS.BN_SIZE,
                'init_features': self.cfg.MODEL.SETTINGS.INIT_FEATURES,
                'growth_rate': self.cfg.MODEL.SETTINGS.GROWTH_RATE,
            }
        )

    def forward(self, x):
        output_list = list()
        for name, module in self.net.features._modules.items():
            x = module(x)
            if isinstance(module, MarkerLayer):
                output_list.append(x)
        return output_list


class Decoder(nn.Module):
    def __init__(self, cfg, block_features):
        super().__init__()
        self.cfg = cfg.clone()
        self.block_features = reversed(block_features)
        self.dilation = self.cfg.MODEL.SETTINGS.DECODER_DILATION

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'decoder_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        in_channels = None
        for i, out_channels in enumerate(self.block_features):
            if i:
                self.add_module(
                    self.get_layer_name(i-1, 'upsampling'),
                    ParamUpSamplingBlock(in_channels, out_channels, scale_factor=(2, 2, 2)) if not i == 1 else
                    LocalizationBlock(in_channels, out_channels),
                )
                self.add_module(
                    self.get_layer_name(i-1, 'localization'),
                    LocalizationBlock(2 * out_channels, out_channels, dilation=self.dilation),
                )
            in_channels = out_channels

    def forward(self, x_input):
        outputs = list()
        x = None
        for i, x_input_i in enumerate(reversed(x_input)):
            if not i:
                x = x_input_i
                continue
            x = getattr(self, self.get_layer_name(i-1, 'upsampling'))(x)
            x, x_input_i = pad_if_necessary(x, x_input_i)
            x = torch.cat((x, x_input_i), dim=1)
            x = getattr(self, self.get_layer_name(i-1, 'localization'))(x)
            if len(x_input)-1-i < self.cfg.MODEL.NUM_PRED_LEVELS:
                outputs.append(x)

        return outputs


class SegHead(nn.Module):
    def __init__(self, cfg, block_features):
        super().__init__()
        self.cfg = cfg.clone()
        self.block_features = list(reversed(block_features))

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'seghead_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        self.num_pred_levels = self.cfg.MODEL.NUM_PRED_LEVELS
        assert self.num_pred_levels < len(self.block_features)
        beg_ind = len(self.block_features) - self.num_pred_levels
        for i in range(self.num_pred_levels):
            in_channels = self.block_features[beg_ind+i]
            self.add_module(
                self.get_layer_name(i, 'conv'),
                nn.Conv3d(in_channels, self.cfg.MODEL.NUM_CLASSES, kernel_size=is_3d((1, 1, 1), IS_3D)),
            )
            self.add_module(
                self.get_layer_name(i, 'upsam'),
                nn.Upsample(scale_factor=is_3d((2, 2, 2), IS_3D), mode='trilinear', align_corners=False),
            )

        self.add_module(
            'upsam_final',
            nn.Upsample(scale_factor=is_3d((2, 2, 2), IS_3D), mode='trilinear', align_corners=False),
        )

    def forward(self, x_input):
        x = 0
        for i in range(self.num_pred_levels):
            x_b = getattr(self, self.get_layer_name(i, 'conv'))(x_input[i])
            if not i == 0:
                x, x_b = pad_if_necessary(x, x_b)
            x = x + x_b
            if not i == self.num_pred_levels - 1:
                x = getattr(self, self.get_layer_name(i, 'upsam'))(x)

        x = getattr(self, 'upsam_final')(x)

        return x


@MODEL_REGISTRY.register()
class Densenet3D(NetABC):
    def __init__(self, cfg):
        super().__init__(cfg, auto_init=False)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

    def _create_net(self):
        self.Encoder = Encoder(self.cfg)
        block_features = self.Encoder.net.block_features
        self.Decoder = Decoder(self.cfg, block_features)
        self.SegHead = SegHead(self.cfg, block_features)

    def forward_core(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.SegHead(x)

        return x
