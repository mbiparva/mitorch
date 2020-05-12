#!/usr/bin/env python3

"""Loss construction functions."""

import torch
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for loss modules.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


# noinspection PyCallingNonCallable
def build_loss(cfg):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the backbone.
    """
    # Construct the loss
    name = cfg.MODEL.LOSS_FUNC
    ignore_index = cfg.MODEL.IGNORE_INDEX
    loss = LOSS_REGISTRY.get(name)(ignore_index=ignore_index)
    return loss