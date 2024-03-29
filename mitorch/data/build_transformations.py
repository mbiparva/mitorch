#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from fvcore.common.registry import Registry

TRANSFORMATION_REGISTRY = Registry("TRANSFORMATIONS")
TRANSFORMATION_REGISTRY.__doc__ = """
Registry for transformations.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


# noinspection PyCallingNonCallable
def build_transformations(dataset_name, cfg, mode):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        mode (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
        cfg (CfgNode): configs.
    Returns:
        Dataset: a constructed transformation list specified by dataset_name.
    """
    if dataset_name in ('SRIBILhfb', 'SRIBILhfbTest', 'LEDUCQTest', 'PPMITest'):
        name = 'HFBTransformations'
    elif dataset_name in ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILTest'):
        name = 'WMHTransformations'
    elif dataset_name in ('WMHSkullStrippingTransformations', ):
        name = 'WMHSkullStrippingTransformations'
    elif dataset_name in ('TRAP', 'CAPTURE', 'TRACING', 'TRACINGSEG'):
        name = 'NVTTransformations'
    elif dataset_name in ('HPSubfield', ):
        name = 'HPSFTransformations'
    else:
        raise NotImplementedError
    return TRANSFORMATION_REGISTRY.get(name)(cfg, mode)
