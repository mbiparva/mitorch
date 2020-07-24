#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from fvcore.common.registry import Registry

TRANSFORMATION_REGISTRY = Registry("TRANSFORMATIONS")
TRANSFORMATION_REGISTRY.__doc__ = """
Registry for transformations.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


# noinspection PyCallingNonCallable
def build_transformations(dataset_name, cfg):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs.
    Returns:
        Dataset: a constructed transformation list specified by dataset_name.
    """
    if dataset_name in ('SRIBILhfb', 'SRIBILhfbTest', 'LEDUCQTest', 'PPMITest'):
        name = 'HFBTransformations'
    elif dataset_name in ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILTest'):
        name = 'WMHTransformations'
    elif dataset_name in ('TRAP', 'CAPTURE'):
        name = 'NVTTransformations'
    else:
        raise NotImplementedError
    return TRANSFORMATION_REGISTRY.get(name)(cfg)
