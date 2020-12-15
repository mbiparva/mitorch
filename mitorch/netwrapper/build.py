#!/usr/bin/env python3

"""Loss construction functions."""

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch.nn as nn
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for loss modules.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


class LossWithLogits(nn.Module):
    def __init__(self, loss) -> None:
        super().__init__()

        self._create_net(loss)

    def _create_net(self, loss):
        self.sigmoid = nn.Sigmoid()
        self.loss = loss

    def forward(self, p, a):
        p = self.sigmoid(p)
        return self.loss(p, a)


# noinspection PyCallingNonCallable
def build_loss(cfg, name, with_logits=True):
    """
    Builds the video model.
    Args:
        with_logits: whether the loss is with logits; if true add sigmoid at the beginning.
        cfg (configs): configs that contains the hyper-parameters to build the backbone.
        name (str): name
    """
    # Construct the loss
    ignore_index = cfg.MODEL.IGNORE_INDEX
    if name == 'WeightedHausdorffLoss':
        loss_params = {
            'whl_num_depth_sheets': cfg.MODEL.WHL_NUM_DEPTH_SHEETS,
            'whl_seg_thr': cfg.MODEL.WHL_SEG_THR,
        }
    elif name == 'FocalLoss':
        loss_params = {
            'alpha': 0.25,
            'gamma': 2.0,
            'reduction': 'mean',
        }
    else:
        loss_params = dict()

    loss = LOSS_REGISTRY.get(name)(ignore_index=ignore_index, **loss_params)

    if with_logits:
        loss = LossWithLogits(loss)

    return loss
