#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_REGISTRY
from .weight_init_helper import init_weights


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1),
                 normalization='instancenorm', nonlinearity='leakyrelu'):
        super().__init__(
            self._create_convolution(in_channels, out_channels, kernel_size, stride, dilation),
            self._create_normalization(out_channels, normalization),
            self._create_nonlinearity(nonlinearity),
        )

    @staticmethod
    def _create_convolution(in_channels, out_channels, kernel_size, stride, dilation):
        return nn.Conv3d(in_channels, out_channels,
                         kernel_size=(3, 3, 3), stride=stride, padding=dilation, dilation=dilation,
                         groups=1, bias=False)  # TODO check bias=True, most nets use False though because of BN

    @staticmethod
    def _create_normalization(out_channels, normalization):
        if normalization is None:
            output = list()
        if normalization == 'batchnorm':
            output = nn.BatchNorm3d(out_channels, momentum=0.1, affine=True, track_running_stats=False, eps=1e-5)
        elif normalization == 'instancenorm':
            output = nn.InstanceNorm3d(out_channels, momentum=0.1, affine=True, track_running_stats=False, eps=1e-5)
        else:
            raise NotImplementedError('only batchnorm, instancenorm, and None are addressed.')

        return output

    @staticmethod
    def _create_nonlinearity(nonlinearity):
        if nonlinearity is None:
            output = list()
        if nonlinearity == 'relu':
            output = nn.ReLU(inplace=True)
        elif nonlinearity == 'leakyrelu':
            output = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError('only relu, leakyrelu and None are addressed.')

        return output


class ContextBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=(3, 3, 3), p=0.3):
        super().__init__(
            BasicBlock(in_channels, out_channels, dilation=dilation),
            nn.Dropout3d(p=p),
            BasicBlock(in_channels, out_channels, dilation=dilation),
        )


class ParamUpSamplingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__(
            nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
            BasicBlock(in_channels, out_channels),
        )


class LocalizationBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__(
            BasicBlock(in_channels, out_channels, dilation=(2, 2, 2)),
            BasicBlock(in_channels, out_channels, kernel_size=(1, 1, 1)),
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

    @staticmethod
    def get_layer_name(i, postfix):
        return 'decoder_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        self.enco_depth = self.cfg.MODEL.ENCO_DEPTH-1  # the for loop begins from 0 ends at d-1
        in_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** self.enco_depth
        for i in reversed(range(self.enco_depth)):
            out_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i
            self.add_module(
                self.get_layer_name(self.enco_depth-1-i, 'upsampling'),
                ParamUpSamplingBlock(in_channels, out_channels, scale_factor=(2, 2, 2)),
            )
            self.add_module(
                self.get_layer_name(self.enco_depth-1-i, 'localization'),
                ParamUpSamplingBlock(2*out_channels, out_channels, scale_factor=(2, 2, 2)),
            )
            in_channels = out_channels

    def forward(self, xs):
        x = xs[self.enco_depth]
        outputs = list()
        for i in reversed(range(self.enco_depth)):
            x = getattr(self, self.get_layer_name(i, 'upsampling'))(x)
            x = torch.cat((x, xs[i]), dim=1)
            x = getattr(self, self.get_layer_name(i, 'localization'))(x)
            if i < self.cfg.MODEL.NUM_PRED_LEVELS:
                outputs.append(x)

        return reversed(outputs)


class SegHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'seghead_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        self.num_pred_levels = self.cfg.MODEL.NUM_PRED_LEVELS-1
        for i in reversed(range(self.num_pred_levels)):
            in_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i
            layer_num = self.num_pred_levels-1-i
            self.add_module(
                self.get_layer_name(layer_num, 'conv'),
                nn.Conv3d(in_channels, self.cfg.MODEL.NUM_CLASSES, kernel_size=(1, 1, 1)),
            )
            if not i:
                self.add_module(
                    self.get_layer_name(layer_num, 'upsam'),
                    nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
                )

    def forward(self, xs):
        x = 0
        for i in range(self.enco_depth):
            x = x + getattr(self, self.get_layer_name(i, 'conv'))(xs[i])
            if not i:
                x = getattr(self, self.get_layer_name(i, 'upsam'))(x)
        return x


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
        self.SegHead = SegHead(self.cfg)

    def init_weights(self):
        init_weights(self, self.cfg.MODEL.FC_INIT_STD)

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.SegHead(x)
        return x
