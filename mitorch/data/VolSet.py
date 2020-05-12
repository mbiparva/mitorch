#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import os
from .build import DATASET_REGISTRY
import torch
import numpy as np
import torch.utils.data as data
import SimpleITK as sitk
import re
from torch._six import container_abcs
from abc import ABC, abstractmethod

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, container_abcs.Mapping):  # this is called for the meta
        return batch
    elif isinstance(elem, container_abcs.Sequence):  # this is called at the beginning
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def read_nii_file(file_path):
    return sitk.ReadImage(file_path)


def extract_meta(data_nii):
    return {
        'origin': data_nii.GetOrigin(),
        'size': data_nii.GetSize(),
        'spacing': data_nii.GetSpacing(),
        'direction': data_nii.GetDirection(),
        'dimension': data_nii.GetDimension(),
        'bitpixel': data_nii.GetMetaData('bitpix'),
    }


class VolSetABC(ABC, data.Dataset):
    def __init__(self, cfg, mode, transform):
        self._init_pars(cfg, mode, transform)
        self._init_dataset()
        self.sample_path_list = sorted(self.sample_path_list)

    def _init_pars(self, cfg, mode, transform):
        self.cfg = cfg
        self.mode = mode
        self.transform = transform
        self.sample_path_list = None
        self.dataset_root = os.path.join(self.cfg.PROJECT.DATASET_DIR, self.__class__.__name__)

    @abstractmethod
    def _init_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def index_samples(self):
        raise NotImplementedError

    @abstractmethod
    def find_data_files_path(self, sample_path):
        raise NotImplementedError

    @staticmethod
    def load_data(t1_path, fl_path, annot_path):
        return (
            read_nii_file(t1_path),
            read_nii_file(fl_path),
            read_nii_file(annot_path),
        )

    @staticmethod
    def extract_data_meta(t1_nii, fl_nii, annot_nii):
        return (
            extract_meta(t1_nii),
            extract_meta(fl_nii),
            extract_meta(annot_nii),
        )

    @staticmethod
    def run_sanity_checks(t1_meta, fl_meta, annot_meta):
        for k in t1_meta.keys():
            t1_meta_k, fl_meta_k, annot_meta_k = t1_meta[k], fl_meta[k], annot_meta[k]
            if k in ('bitpixel', ):
                continue
            try:
                match = t1_meta_k == fl_meta_k == annot_meta_k
                assert match, '{} does not match in all three: \n{}\n{}\n{}\nrounding is going to be started'.format(
                    k,
                    t1_meta_k,
                    fl_meta_k,
                    annot_meta_k
                )
            except AssertionError:
                ndigits = 8 - 2
                while ndigits >= 2:
                    t1_meta_k = [round(i, ndigits) for i in t1_meta_k]
                    fl_meta_k = [round(i, ndigits) for i in fl_meta_k]
                    annot_meta_k = [round(i, ndigits) for i in annot_meta_k]
                    if t1_meta_k == fl_meta_k == annot_meta_k:
                        break
                    ndigits -= 2
                if ndigits == 0:
                    raise Exception(
                        '{} does not match in all three: \n{}\n{}\n{}'.format(
                            k,
                            t1_meta_k,
                            fl_meta_k,
                            annot_meta_k
                        )
                    )

        return t1_meta  # we keep only one of them

    @staticmethod
    @abstractmethod
    def curate_annotation(annot_tensor, ignore_index):
        raise NotImplementedError

    def get_data_tensor(self, t1_nii, fl_nii, annot_nii):
        t1_tensor, fl_tensor, annot_tensor = (
            torch.tensor(
                data=sitk.GetArrayFromImage(t1_nii).astype(np.float32),
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            ),
            torch.tensor(
                data=sitk.GetArrayFromImage(fl_nii).astype(np.float32),
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            ),
            torch.tensor(
                data=sitk.GetArrayFromImage(annot_nii).astype(np.float32),
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            )
        )

        annot_tensor = self.curate_annotation(annot_tensor, ignore_index=self.cfg.MODEL.IGNORE_INDEX)

        return t1_tensor, fl_tensor, annot_tensor

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_path_list)
