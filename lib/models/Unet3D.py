#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_REGISTRY
from .weight_init_helper import init_weights


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), dilation=(1, 1, 1)):
        # TODO this can be more fancier, with residual connections
        super().__init__(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=(3, 3, 3), stride=stride, padding=dilation, dilation=dilation,
                      groups=1, bias=False),  # TODO check bias=True, most nets use False though because of BN
            nn.InstanceNorm3d(out_channels, momentum=0.1, affine=True, track_running_stats=False, eps=1e-5),
            nn.ReLU(inplace=True)  # TODO try nn.LeakyReLU as the TF code suggests
        )


class ContextBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=(3, 3, 3), p=0.3):
        super().__init__(
            BasicBlock(in_channels, out_channels, dilation=dilation),
            nn.Dropout3d(p=p),
            BasicBlock(in_channels, out_channels, dilation=dilation),
        )


class CompoundBlock(nn.Module):
    def __init__(self, i, in_channels, out_channels, stride, dilation, p):
        super().__init__()

        self._create_net(i, in_channels, out_channels, stride, dilation, p)

    def _create_net(self, i, in_channels, out_channels, stride, dilation, p):
        kwargs = dict()
        if i == 0:
            kwargs['dilation'] = dilation
        else:
            kwargs['stride'] = stride
        self.input_layer = BasicBlock(in_channels, out_channels, **kwargs)
        self.context_layer = ContextBlock(in_channels, out_channels, dilation=dilation, p=p)

    def forward(self, x):
        x_input = self.input_layer(x)
        x = self.context_layer(x_input)
        x = x_input + x
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stride = self.dilation = (2, 2, 2)
        self.p = 0.3

        self._create_net()

    @staticmethod
    def get_layer_name(i):
        return 'encoder_layer{:03}'.format(i)

    def _create_net(self):
        in_channels = self.cfg.MODEL.INPUT_CHANNELS
        for i in range(self.cfg.MODEL.ENCO_DEPTH):
            out_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i
            self.add_module(
                self.get_layer_name(i),
                CompoundBlock(i, in_channels, out_channels, stride=self.stride, dilation=self.dilation, p=self.p),
            )
            in_channels = out_channels

    def forward(self, x):
        outputs = list()
        for i in range(self.cfg.MODEL.ENCO_DEPTH):
            x = getattr(self, self.get_layer_name(i))(x)
            outputs.append(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net()

    def _create_net(self):
        # TODO NEXT CONTINUE FRMO HERE
        pass


@MODEL_REGISTRY.register()
class Unet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net()

        self.init_weights()

    def _create_net(self):
        self.Encoder = Encoder(self.cfg)
        self.Decoder = Decoder(self.cfg)

    def init_weights(self):
        init_weights(self, self.cfg.MODEL.FC_INIT_STD)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x
