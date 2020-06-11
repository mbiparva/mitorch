#!/usr/bin/env python3
#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""Utility function for weight initialization"""

import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill


def init_weights(model, fc_init_std=0.01):
    """
    Performs ResNet style weight initialization.
    Args:
        model (nn.Module): the network containing the parameters and the forward function
        fc_init_std (float): the expected standard deviation for fc layer.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):  # This assumes nn.Linear is the final layers
            # TODO check to see if this is effective in this architecture since the final is a conv3d
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()
