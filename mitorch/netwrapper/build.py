#!/usr/bin/env python3

"""Loss construction functions."""

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for loss modules.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


# noinspection PyCallingNonCallable
def build_loss(cfg, name=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the backbone.
        name (str): name
    """
    # Construct the loss
    name = cfg.MODEL.LOSS_FUNC if name is None else name
    ignore_index = cfg.MODEL.IGNORE_INDEX
    if name == 'WeightedHausdorffLoss':
        loss_params = {
            'whl_num_depth_sheets': cfg.MODEL.WHL_NUM_DEPTH_SHEETS,
            'whl_seg_thr': cfg.MODEL.WHL_SEG_THR,
        }
    elif name == 'FocalLoss':
        loss_params = {
            'alpha': 1.0,
            'gamma': 2.0,
            'reduction': 'mean',
        }
    else:
        loss_params = dict()
    loss = LOSS_REGISTRY.get(name)(ignore_index=ignore_index, **loss_params)
    return loss
