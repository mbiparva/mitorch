#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn
from .build import MODEL_REGISTRY
from .weight_init_helper import init_weights
from utils.models import pad_if_necessary

IS_3D = True


def is_3d(size, lb=1):
    if isinstance(size, (tuple, list)):
        assert len(size) == 3, 'expects 3D iterables'
        return (
            is_3d(size[0], lb=lb),
            size[1],
            size[2],
        )
    elif isinstance(size, (int, float)):
        return size if IS_3D else lb
    else:
        raise NotImplementedError


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1),
            normalization=('batchnorm', 'instancenorm')[1], nonlinearity=('relu', 'leakyrelu')[1]
    ):
        super().__init__(
            self._create_convolution(in_channels, out_channels, kernel_size, stride, dilation),
            self._create_normalization(out_channels, normalization),
            self._create_nonlinearity(nonlinearity),
        )

    @staticmethod
    def _create_convolution(in_channels, out_channels, kernel_size, stride, dilation):
        padding = tuple((torch.tensor(kernel_size) // 2 + torch.tensor(dilation) // 2).tolist())  # does the same
        return nn.Conv3d(
            in_channels, out_channels, kernel_size=is_3d(kernel_size), stride=is_3d(stride),
            padding=is_3d(padding, lb=0), dilation=is_3d(padding), groups=1, bias=False
        )  # TODO check bias=True, most nets use False though because of BN

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
            BasicBlock(out_channels, out_channels, dilation=dilation),
        )


class ParamUpSamplingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super().__init__(
            nn.Upsample(scale_factor=is_3d(scale_factor), mode='trilinear', align_corners=False),
            BasicBlock(in_channels, out_channels),
        )


class LocalizationBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=(2, 2, 2)):
        super().__init__(
            BasicBlock(in_channels, out_channels, dilation=dilation),
            BasicBlock(out_channels, out_channels, kernel_size=(3, 3, 3)),  # network architecture changed
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
        self.context_layer = ContextBlock(out_channels, out_channels, dilation=dilation, p=p)

    def forward(self, x):
        x_input = self.input_layer(x)
        x = self.context_layer(x_input)
        x = x_input + x
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.stride = (2, 2, 2)
        self.dilation = (2, 2, 2)
        self.p = self.cfg.MODEL.DROPOUT_RATE

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
        self.cfg = cfg.clone()

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'decoder_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        self.enco_depth = self.cfg.MODEL.ENCO_DEPTH - 1  # the for loop begins from 0 ends at d-1
        in_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** self.enco_depth
        for i in range(self.enco_depth):
            i_r = self.enco_depth - 1 - i
            out_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i_r
            self.add_module(
                self.get_layer_name(i, 'upsampling'),
                ParamUpSamplingBlock(in_channels, out_channels, scale_factor=(2, 2, 2)),
            )
            self.add_module(
                self.get_layer_name(i, 'localization'),
                LocalizationBlock(2 * out_channels, out_channels),
            )
            in_channels = out_channels

    def forward(self, x_input):
        x = x_input[self.enco_depth]
        outputs = list()
        for i in range(self.enco_depth):
            i_r = self.enco_depth - 1 - i
            x = getattr(self, self.get_layer_name(i, 'upsampling'))(x)
            x, x_input[i_r] = pad_if_necessary(x, x_input[i_r])
            x = torch.cat((x, x_input[i_r]), dim=1)
            x = getattr(self, self.get_layer_name(i, 'localization'))(x)
            if i_r < self.cfg.MODEL.NUM_PRED_LEVELS:
                outputs.append(x)

        return outputs


class SegHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'seghead_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        self.num_pred_levels = self.cfg.MODEL.NUM_PRED_LEVELS
        for i in range(self.num_pred_levels):
            i_r = self.num_pred_levels - 1 - i
            in_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i_r
            self.add_module(
                self.get_layer_name(i, 'conv'),
                nn.Conv3d(in_channels, self.cfg.MODEL.NUM_CLASSES, kernel_size=is_3d((1, 1, 1))),
            )
            if not i == self.num_pred_levels - 1:
                self.add_module(
                    self.get_layer_name(i, 'upsam'),
                    nn.Upsample(scale_factor=is_3d((2, 2, 2)), mode='trilinear', align_corners=False),
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

        if self.cfg.MODEL.LOSS_FUNC in ('DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss'):
            x = nn.Sigmoid()(x)

        return x


@MODEL_REGISTRY.register()
class Unet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = self.set_model_settings(cfg.clone())

        self.set_processing_mode()

        self._create_net()

        self.init_weights()

    @staticmethod
    def set_model_settings(cfg):
        if 'SETTINGS' in cfg.MODEL and isinstance(cfg.MODEL.SETTINGS, tuple):
            cfg_MODEL_SETTINGS = dict(cfg.MODEL.SETTINGS)
            cfg.MODEL.SETTINGS = cfg_MODEL_SETTINGS[cfg.MODEL.MODEL_NAME] \
                if cfg.MODEL.MODEL_NAME in cfg_MODEL_SETTINGS else cfg_MODEL_SETTINGS

        return cfg

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'

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
