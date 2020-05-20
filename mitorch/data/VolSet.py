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
# import SimpleITK as sitk
import nibabel as nib
import re
from torch._six import container_abcs
from abc import ABC, abstractmethod
import data.utils_ext as utils_ext


np_str_obj_array_pattern = re.compile(r'[SaUO]')
# RAI codes
# c3d's default: (('R', 'L'), ('A', 'P'), ('I', 'S'))
# nib's default: (('L', 'R'), ('P', 'A'), ('I', 'S'))
c3d_labels = (('R', 'L'), ('A', 'P'), ('S', 'I'))


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


# noinspection PyUnresolvedReferences
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


def read_nii_file(file_path, **kwargs):
    return nib_loader(file_path, **kwargs)


# ----------- START RENAMING ---------------------

def nib_loader(filename, enforce_nib_canonical=False, enforce_diag=False, dtype=np.float32):
    vol = nib.load(filename)
    vol = utils_ext.correct_nifti_header_if_necessary(vol)
    header = dict(vol.header)
    header["filename"] = filename
    header["affine"] = vol.affine
    header["original_affine"] = vol.affine.copy()
    header["enforce_nib_canonical"] = enforce_nib_canonical

    header["origin"] = vol.affine[:, -1][:3]
    header["size"] = vol.shape
    header["spacing"] = vol.header.get_zooms()
    header["dimension"] = vol.header["dim"][0]
    header["bitpix"] = int(header["bitpix"])

    dim = header["dimension"]
    RZS = vol.affine[:dim, :dim]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    zooms[zooms == 0] = 1
    header["direction"] = list((RZS / zooms).flatten())

    if enforce_nib_canonical:
        vol = nib.as_closest_canonical(vol, enforce_diag=enforce_diag)
        header["affine"] = vol.affine

    img_array = np.array(vol.get_fdata(dtype=dtype))
    vol.uncache()

    assert img_array.shape == header['size']

    return img_array, header

# ----------- END RENAMING ---------------------


def np2tuple(in_array):
    return tuple(in_array.flatten()) if isinstance(in_array, np.ndarray) else in_array


def extract_meta(in_meta):
    return {
        'origin': np2tuple(in_meta['origin']),
        'size': np2tuple(in_meta['size']),
        'spacing': np2tuple(in_meta['spacing']),
        'direction': np2tuple(in_meta['direction']),
        'dimension': np2tuple(in_meta['dimension']),
        'bitpixel': np2tuple(in_meta['bitpix']),
        'affine': np2tuple(in_meta['affine'])
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
    def load_data(in_pipe, **kwargs):
        return {
            u: read_nii_file(v, **kwargs)
            for u, v in in_pipe.items()
        }

    @staticmethod
    def extract_data_meta(in_pipe):
        in_pipe_data, in_pipe_meta = dict(), dict()
        for u, (d, m) in in_pipe.items():
            in_pipe_data[u] = d
            in_pipe_meta[u] = extract_meta(m)

        return (
            in_pipe_data,
            in_pipe_meta
        )

    @staticmethod
    def run_sanity_checks(in_pipe_meta):
        first_item = in_pipe_meta[list(in_pipe_meta.keys())[0]]
        for k in first_item.keys():
            if k in ('bitpixel', ):
                continue
            in_pipe_meta_value = [v[k] for v in in_pipe_meta.values()]
            ndigits = 8
            while ndigits >= 2:
                in_pipe_meta_value_rounded = [
                    [
                        round(j, ndigits)
                        for j in i
                    ]
                    if isinstance(i, (list, tuple)) else round(i, ndigits)
                    for i in in_pipe_meta_value
                ]
                first_pipe = in_pipe_meta_value_rounded[0]
                all_equal = True
                for i in in_pipe_meta_value_rounded[1:]:
                    all_equal = all_equal and (first_pipe == i)
                if all_equal:
                    break
                ndigits -= 2
            if ndigits == 0:
                raise Exception(
                    '{} does not match in all pipe elements ({})'.format(
                        k,
                        in_pipe_meta_value
                    )
                )

        return first_item  # Send out the first one

    @staticmethod
    @abstractmethod
    def curate_annotation(annot_tensor, ignore_index):
        raise NotImplementedError

    def get_data_tensor(self, in_pipe_data):
        in_pipe_data = {
            u: torch.tensor(
                data=v,
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            )
            for u, v in in_pipe_data.items()
        }

        annot_tensor = in_pipe_data.pop('annot')
        image_tensor = list(in_pipe_data.values())

        annot_tensor = self.curate_annotation(annot_tensor, ignore_index=self.cfg.MODEL.IGNORE_INDEX)

        return image_tensor, annot_tensor

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        in_pipe_data = self.find_data_files_path(sample_path)
        in_pipe_data = self.load_data(in_pipe_data, enforce_nib_canonical=False, enforce_diag=False, dtype=np.float32)
        in_pipe_data, in_pipe_meta = self.extract_data_meta(in_pipe_data)

        in_pipe_meta = self.run_sanity_checks(in_pipe_meta)
        in_pipe_meta['sample_path'] = sample_path

        image_tensor, annot_tensor = self.get_data_tensor(in_pipe_data)

        image_tensor = torch.stack(image_tensor, dim=-1)  # D x H x W x C
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, annot_tensor, in_pipe_meta = self.transform((image_tensor, annot_tensor, in_pipe_meta))

        return image_tensor, annot_tensor, in_pipe_meta

    def __len__(self):
        return len(self.sample_path_list)
