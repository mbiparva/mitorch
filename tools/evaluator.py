#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from batch_abc import BatchBase
import torch


class Evaluator(BatchBase):
    def __init__(self, cfg, device):
        super().__init__('test' if cfg.TEST.ENABLE else 'valid', cfg, device)

    def set_net_mode(self, net):
        net.eval()
