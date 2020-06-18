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
        super().__init__(cfg, mode, transform)
        self.prefix_name = True

    def index_samples(self):
        return [
            os.path.join(self.dataset_path, i)
            for i in sorted(os.listdir(self.dataset_path))
        ]

    def put_fname_template(self, path_name, file_name):
        prefix = '{}_'.format(path_name.rpartition('/')[-1]) if self.prefix_name else ''
        return '{}{}'.format(
            prefix,
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
        self.prefix_name = False

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


@DATASET_REGISTRY.register()
class SRIBILhfbTest(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)
        self.prefix_name = True


@DATASET_REGISTRY.register()
class LEDUCQTest(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)
        self.prefix_name = True


@DATASET_REGISTRY.register()
class PPMITest(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)
        self.prefix_name = True

    @staticmethod
    def run_sanity_checks(in_pipe_meta):
        return in_pipe_meta['fl']  # fl is correct, we skip sanity check since they don't match

    # The reason we override getitem here in PPMI is that there is a misalignment between Flair and other modalities
    # Flair has shape (x, x, x) others have (x, x, x, 1) which is wrong
    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        in_pipe_data = self.find_data_files_path(sample_path)
        in_pipe_data = self.load_data(in_pipe_data,
                                      enforce_nib_canonical=self.cfg.DATA.ENFORCE_NIB_CANONICAL,
                                      enforce_diag=self.cfg.DATA.ENFORCE_DIAG,
                                      dtype=np.float32)
        in_pipe_data, in_pipe_meta = self.extract_data_meta(in_pipe_data)
        for u, v in in_pipe_data.items():
            if v.ndim == 4:
                assert v.shape[-1] == 1
                in_pipe_data[u] = in_pipe_data[u].squeeze(-1)

        in_pipe_meta = self.run_sanity_checks(in_pipe_meta)
        in_pipe_meta['sample_path'] = sample_path

        image_tensor, annot_tensor = self.get_data_tensor(in_pipe_data)

        image_tensor = torch.stack(image_tensor, dim=-1)  # D x H x W x C
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, annot_tensor, in_pipe_meta = self.transform((image_tensor, annot_tensor, in_pipe_meta))

        return image_tensor, annot_tensor, in_pipe_meta
