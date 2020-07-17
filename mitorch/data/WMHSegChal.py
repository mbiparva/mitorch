#!/usr/bin/env python3

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
import SimpleITK as sitk
import re
from torch._six import container_abcs
from .VolSet import VolSetABC


@DATASET_REGISTRY.register()
class WMHSegmentationChallenge(VolSetABC):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = os.path.join(self.dataset_root, 'uncompressed')
        self.in_modalities = self.cfg.TEST.IN_MOD if self.mode == 'test' else self.cfg.TRAIN.IN_MOD
        self.in_modalities = {u: v for u, v in self.in_modalities}
        self.sample_path_list = self.index_samples()

    def index_samples(self):
        screening_sites = sorted(os.listdir(self.dataset_path))
        assert len(screening_sites) == 3, 'expects 3 screening sites folders'
        assert screening_sites == sorted(('GE3T', 'Singapore', 'Utrecht')), 'expects to have only 3 folders'

        return [
            os.path.join(self.dataset_path, f, i)
            for f in screening_sites
            for i in sorted(os.listdir(os.path.join(self.dataset_path, f)))
        ]

    def find_data_files_path(self, sample_path):
        return {
            u: os.path.join(sample_path, 'pre', v)
            if u not in ('annot', ) else os.path.join(sample_path, v)
            for u, v in self.in_modalities.items()
        }

    @staticmethod
    def curate_annotation(annot_tensor, ignore_index):
        cat_labels = set(annot_tensor.unique(sorted=True).tolist())
        known_labels = set(tuple([0, 1, 2]))
        assert cat_labels.issubset(known_labels), 'only expect labels of {} in annotations {}'.format(
            known_labels,
            cat_labels
        )
        if 2 in cat_labels:
            annot_tensor[annot_tensor == 2] = ignore_index
            # annot_tensor[annot_tensor == 2] = 0
        return annot_tensor
