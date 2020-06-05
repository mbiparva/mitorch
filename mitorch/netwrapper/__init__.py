#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from .build import LOSS_REGISTRY, build_loss
from .loss import CrossEntropyLoss, DiceLoss, WeightedHausdorffLoss, AveragedHausdorffLoss
