#!/usr/bin/env python3

"""Model construction functions."""

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
from fvcore.common.registry import Registry
from torch import nn
import utils.logging as logging

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


# noinspection PyCallingNonCallable
def build_model(cfg, cur_device):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the backbone.
        cur_device (int): select the GPU id to load the model to its memory
    """
    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.USE_GPU:
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    else:
        logger.info('Using CPU for the network operations')

    if torch.cuda.device_count() > 1 and cfg.DP:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        print(model.device_ids)
    elif cfg.DDP:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )

    return model
