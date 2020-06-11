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


# noinspection PyBroadException
@DATASET_REGISTRY.register()
class SRIBIL(VolSetABC):
    def __init__(self, cfg, mode, transform):
        self.hfb_annot = cfg.TRAIN.SRIBIL_HFB_ANNOT
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = self.dataset_root
        self.in_modalities = {  # TODO this could be passed as an input argument or config attribute based on users need
            't1': 'T1_nu.nii.gz',
            'fl': 'T1acq_nu_FL.nii.gz',
            'annot': ('wmh_seg.nii.gz', 'T1acq_nu_HfBd.nii.gz')[self.hfb_annot is True],
            # 't2': None,  # Add any more modalities you want HERE
        }
        self.sample_path_list = self.index_samples()

    def index_samples(self):
        return [
            os.path.join(self.dataset_path, i)
            for i in sorted(os.listdir(self.dataset_path))
        ]

    @staticmethod
    def put_fname_template(path_name, file_name):
        par_dir = path_name.rpartition('/')[-1]
        return '{}_{}'.format(
            par_dir,
            file_name
        )

    def find_data_files_path(self, sample_path):
        return {
            u: os.path.join(sample_path, self.put_fname_template(sample_path, v))
            for u, v in self.in_modalities.items()
        }

    @staticmethod
    def curate_annotation(annot_tensor, ignore_index):
        cat_labels = set(annot_tensor.unique(sorted=True).tolist())
        known_labels = set(tuple([0, 1]))
        assert cat_labels.issubset(known_labels), 'only expect labels of {} in annotations {}'.format(
            known_labels,
            cat_labels
        )
        # if 2 in cat_labels:
        #     annot_tensor[annot_tensor == 2] = ignore_index
        return annot_tensor


@DATASET_REGISTRY.register()
class SRIBILhfb(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = self.dataset_root
        self.in_modalities = {  # TODO this could be passed as an input argument or config attribute based on users need
            't1': 't1.nii.gz',
            'fl': 'flair.nii.gz',
            'annot': 'truth.nii.gz',
            # 't2': None,  # Add anymore modalities you want HERE
        }
        self.sample_path_list = self.index_samples()

    def find_data_files_path(self, sample_path):
        return {
            u: os.path.join(sample_path, v)
            for u, v in self.in_modalities.items()
        }

    @staticmethod
    def curate_annotation(annot_tensor, ignore_index):
        cat_labels = set(annot_tensor.unique(sorted=True).int().tolist())
        known_labels = set(tuple([0, 1]))
        try:
            assert cat_labels.issubset(known_labels), 'only expect labels of {} in annotations {}'.format(
                known_labels,
                cat_labels
            )
        except AssertionError as e:
            print('out of label set assertion caught ... known_labels variable is revised with the second option.')
            known_labels = set(tuple([0, 1, 2, 3, 5, 6, 7, 8]))
            assert cat_labels.issubset(known_labels), 'this should never happen. If it does, the dataset is changed.'
            annot_tensor[annot_tensor != 8] = 0  # checked in itk-snap app. Label 8 is the head.
            annot_tensor[annot_tensor == 8] = 1
        return annot_tensor
