#!/usr/bin/env python3
#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""Utility function for weight initialization"""

import torch.nn as nn
# from fvcore.nn.weight_init import c2_msra_fill


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
            c2_msra_fill(m, nonlinearity=('relu', 'leaky_relu')[0])
            # c2_xavier_fill(m)
            # nn.init.xavier_normal_(m.weight)
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:  # pyre-ignore
            #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):  # This assumes nn.Linear is the final layers
            # TODO check to see if this is effective in this architecture since the final is a conv3d
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            m.bias.data.zero_()


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module, nonlinearity: str = "relu") -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
        nonlinearity (str): nonlinearity.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity=nonlinearity)
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)
