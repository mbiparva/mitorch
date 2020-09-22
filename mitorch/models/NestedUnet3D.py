#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn
from .build import MODEL_REGISTRY
from utils.models import pad_if_necessary_all
from models.Unet3D import Unet3D, Encoder as Unet3DEncoder, ParamUpSamplingBlock, LocalizationBlock, is_3d
import models.Unet3D


IS_3D = True


class ModulationBlock(nn.Module):
    def __init__(self, modulation_type):
        super().__init__()
        assert modulation_type in ('additive', 'multiplicative', 'mean', 'concatenation')
        self.modulation_type = modulation_type

    def forward(self, x):
        if self.modulation_type == 'additive':
            x = torch.stack(x, dim=1)
            x = torch.sum(x, dim=1)
        elif self.modulation_type == 'multiplicative':
            x = torch.stack(x, dim=1)
            x = torch.prod(x, dim=1)
        elif self.modulation_type == 'mean':
            x = torch.stack(x, dim=1)
            x = torch.mean(x, dim=1)
        elif self.modulation_type == 'concatenation':
            x = torch.cat(x, dim=1)
        else:
            raise NotImplementedError

        return x


class DeepAggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 modulation_type='concatenation', num_in_modal=2, scale_factor=(2, 2, 2)):
        super().__init__()
        self.modulation_type = modulation_type
        self.num_in_modal = num_in_modal

        self._create_net(in_channels, out_channels, modulation_type, num_in_modal, scale_factor)

    def _create_net(self, in_channels, out_channels, modulation_type, num_in_modal, scale_factor):
        localization_input_multiplier = num_in_modal if modulation_type == 'concatenation' else 1

        self.upsampling = ParamUpSamplingBlock(in_channels, out_channels, scale_factor=scale_factor)
        self.modulation = ModulationBlock(modulation_type)
        self.localization = LocalizationBlock(localization_input_multiplier * out_channels, out_channels)

    def forward(self, x_in):
        assert len(x_in) == self.num_in_modal and isinstance(x_in, (tuple, list))

        x_in_skip, x_in_downsized = x_in[:-1], x_in[-1]  # the last one is the immediate one

        x = self.upsampling(x_in_downsized)
        x_in_skip, x = pad_if_necessary_all(x_in_skip, x)
        x_list = x_in_skip + [x]
        x = self.modulation(x_list)
        x = self.localization(x)

        return x


class Encoder(Unet3DEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

        self._create_net()

    @staticmethod
    def get_layer_number(i, j):
        return '{:03}_{:03}'.format(i, j)

    def get_layer_name(self, i, j, postfix):
        return 'decoder_layer{}_{}'.format(self.get_layer_number(i, j), postfix)

    @staticmethod
    def get_ij_num_in_model(j, n_hop):
        return min(
                    j,
                    n_hop,
                ) + 1  # + 1 is for the downsized input.

    def _create_net(self):
        for i in range(self.cfg.MODEL.ENCO_DEPTH-1):
            out_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** i
            in_channels = out_channels * 2
            for j in range(1, self.cfg.MODEL.ENCO_DEPTH-i):  # 0_j is the encoding layer at depth j
                ij_num_in_modal = self.get_ij_num_in_model(j, self.cfg.MODEL.SETTINGS.N_HOP_DENSE_SKIP_CONNECTION)
                self.add_module(
                    self.get_layer_name(i, j, 'deep_aggregation'),
                    DeepAggregationBlock(in_channels, out_channels,
                                         modulation_type=self.cfg.MODEL.SETTINGS.MODULATION_TYPE,
                                         num_in_modal=ij_num_in_modal, scale_factor=(2, 2, 2)),
                )

    def fetch_in_modal(self, outputs, layer_coord, ij_num_in_modal):
        ij_num_in_modal -= 1  # skip the one from the level below, it is added separately
        i, j = layer_coord
        return [
            outputs[self.get_layer_number(i, j-k-1)]
            for k in reversed(range(ij_num_in_modal))
        ] + [
            outputs[self.get_layer_number(i+1, j-1)]
        ]

    def forward(self, x_input):
        outputs = dict()
        for j in range(self.cfg.MODEL.ENCO_DEPTH):
            for i in range(self.cfg.MODEL.ENCO_DEPTH-j):
                if j == 0:
                    outputs[self.get_layer_number(i, j)] = x_input[i]
                    continue
                ij_num_in_modal = self.get_ij_num_in_model(j, self.cfg.MODEL.SETTINGS.N_HOP_DENSE_SKIP_CONNECTION)
                x = self.fetch_in_modal(outputs, (i, j), ij_num_in_modal)
                x = getattr(self, self.get_layer_name(i, j, 'deep_aggregation'))(x)
                outputs[self.get_layer_number(i, j)] = x

        return {
            j: outputs[self.get_layer_number(0, j)]
            for j in range(1, self.cfg.MODEL.ENCO_DEPTH)
        }  # only return those for predictions


class SegHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

        self._create_net()

    @staticmethod
    def get_layer_name(i, postfix):
        return 'seghead_layer{:03}_{}'.format(i, postfix)

    def _create_net(self):
        in_channels = self.cfg.MODEL.N_BASE_FILTERS * 2 ** 0  # it is always row 0

        self.lower_bound = 1 if self.cfg.MODEL.SETTINGS.DEEP_SUPERVISION else self.cfg.MODEL.ENCO_DEPTH - 1
        for j in range(self.lower_bound, self.cfg.MODEL.ENCO_DEPTH):
            self.add_module(
                self.get_layer_name(j, 'conv'),
                nn.Conv3d(in_channels, self.cfg.MODEL.NUM_CLASSES, kernel_size=is_3d((1, 1, 1))),
            )

    def forward(self, x_input):
        x = list()
        for j in range(self.lower_bound, self.cfg.MODEL.ENCO_DEPTH):
            layer_j = getattr(self, self.get_layer_name(j, 'conv'))
            input_j = x_input[j]

            x_j = layer_j(input_j)

            x.append(x_j)

        if self.cfg.MODEL.LOSS_FUNC in ('DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss'):
            x = [nn.Sigmoid()(i) for i in x]

        return x


@MODEL_REGISTRY.register()
class NestedUnet3D(Unet3D):
    def __init__(self, cfg):
        super().__init__(cfg)

    def set_processing_mode(self):
        global IS_3D
        IS_3D = self.cfg.MODEL.PROCESSING_MODE == '3d'
        super().set_processing_mode()

    def _create_net(self):
        self.Encoder = Encoder(self.cfg)
        self.Decoder = Decoder(self.cfg)
        self.SegHead = SegHead(self.cfg)
