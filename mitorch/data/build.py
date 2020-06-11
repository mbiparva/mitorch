#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


# noinspection PyCallingNonCallable
def build_dataset(dataset_name, cfg, split, transformations):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs.
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
        transformations: the list of all pre-processing transformations
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name
    return DATASET_REGISTRY.get(name)(cfg, split, transformations)
