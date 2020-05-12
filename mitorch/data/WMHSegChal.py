#!/usr/bin/env python3
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
        self.t1_file_name, self.fl_file_name, self.annot_file_name = ('T1.nii.gz', 'FLAIR.nii.gz', 'wmh.nii.gz')
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
        return (
            os.path.join(sample_path, 'pre', self.t1_file_name),
            os.path.join(sample_path, 'pre', self.fl_file_name),
            os.path.join(sample_path, self.annot_file_name),
        )

    @staticmethod
    def curate_annotation(annot_tensor, ignore_index):
        cat_labels = set(annot_tensor.unique(sorted=True).tolist())
        known_labels = set(tuple([0, 1, 2]))
        assert cat_labels.issubset(known_labels), 'only expect labels of {} in annotations {}'.format(
            known_labels,
            cat_labels
        )
        if 2 in cat_labels:
            annot_tensor[annot_tensor == 2] = ignore_index  # TODO check this with Unet3D to see what is done there.
        return annot_tensor

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        t1_path, fl_path, annot_path = self.find_data_files_path(sample_path)
        t1_nii, fl_nii, annot_nii = self.load_data(t1_path, fl_path, annot_path)
        meta_data = self.extract_data_meta(t1_nii, fl_nii, annot_nii)

        meta_data = self.run_sanity_checks(*meta_data)
        meta_data['sample_path'] = sample_path

        t1_tensor, fl_tensor, annot_tensor = self.get_data_tensor(t1_nii, fl_nii, annot_nii)

        image_tensor = torch.stack((t1_tensor, fl_tensor), dim=-1)  # D x H x W x C
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, annot_tensor, meta_data = self.transform((image_tensor, annot_tensor, meta_data))

        return image_tensor, annot_tensor, meta_data
