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


# noinspection PyBroadException
@DATASET_REGISTRY.register()
class SRIBIL(VolSetABC):
    def __init__(self, cfg, mode, transform):
        self.hfb_annot = cfg.TRAIN.SRIBIL_HFB_ANNOT
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = self.dataset_root
        self.t1_file_name, self.fl_file_name = ('T1_nu.nii.gz', 'T1acq_nu_FL.nii.gz')
        self.annot_file_name = ('wmh_seg.nii.gz', 'T1acq_nu_HfBd.nii.gz')[self.hfb_annot is True]
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
        return (
            os.path.join(sample_path, self.put_fname_template(sample_path, self.t1_file_name)),
            os.path.join(sample_path, self.put_fname_template(sample_path, self.fl_file_name)),
            os.path.join(sample_path, self.put_fname_template(sample_path, self.annot_file_name)),
        )

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


@DATASET_REGISTRY.register()
class SRIBILhfb(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = self.dataset_root
        self.t1_file_name, self.fl_file_name = ('t1.nii.gz', 'flair.nii.gz')
        self.annot_file_name = 'truth.nii.gz'
        self.sample_path_list = self.index_samples()

    def find_data_files_path(self, sample_path):
        return (
            os.path.join(sample_path, self.t1_file_name),
            os.path.join(sample_path, self.fl_file_name),
            os.path.join(sample_path, self.annot_file_name),
        )

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
