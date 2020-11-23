#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from test_net_single import test as test_single
from test_net_multiple import test as test_multiple


def test(cfg):
    if cfg.TEST.ROBUST_EXP:
        test_multiple(cfg)
    else:
        test_single(cfg)
