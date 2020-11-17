#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
from .build import DATASET_REGISTRY
import torch
import numpy as np
import torch.utils.data as data
from .SRIBILSet import SRIBILBase
import re


@DATASET_REGISTRY.register()
class HPSubfield(SRIBILBase):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)

    def index_samples(self):
        return [
            os.path.join(self.dataset_path, i)
            for i in sorted(os.listdir(self.dataset_path))
            if re.search(r'^s\d\d$', i) is not None
        ]
