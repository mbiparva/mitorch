#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from fvcore.common.registry import Registry

TESTPIPELINE_REGISTRY = Registry("TESTPIPELINE")
TESTPIPELINE_REGISTRY.__doc__ = """
Registry for test pipeline consisting of a set of predefined transformations.

The registered object will be called with `obj(cfg, expt)`.
The call should return a `list` object.
"""


# noinspection PyCallingNonCallable
def get_test_transformation_name(t):
    name = t['t_name']
    name = f'{name[0].upper()}{name[1:]}'
    name = '{}TestTransformations'.format(name)

    return name


def build_transformations(cfg, exp):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (CfgNode): configs.
        exp (dict): dictionary of experiment transformations
    Returns:
        List of transformations: a constructed transformation list specified by exp.
    """
    output_pipeline = list()
    for t in exp:
        name, params = get_test_transformation_name(t), t['t_params']

        trans = TESTPIPELINE_REGISTRY.get(name)(cfg, params)

        output_pipeline.append(trans)

    return output_pipeline
