#!/usr/bin/env python3

import os
from .build import DATASET_REGISTRY
import torch.utils.data as data


@DATASET_REGISTRY.register()
class WMHSegmentationChallenge(data.Dataset):
    def __init__(self, cfg, transformations):
        self.cfg = cfg,
        self.transformations = transformations
        self.dataset_path = os.path.join(self.cfg.PROJECT.DATASET_DIR, self.__class__.__name__)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
