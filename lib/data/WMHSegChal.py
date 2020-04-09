#!/usr/bin/env python3
import os
from .build import DATASET_REGISTRY
import torch
import numpy as np
import torch.utils.data as data
import SimpleITK as sitk


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


@DATASET_REGISTRY.register()
class WMHSegmentationChallenge(data.Dataset):
    def __init__(self, cfg, transform):
        self.cfg = cfg
        self.transform = transform
        self.dataset_path = os.path.join(self.cfg.PROJECT.DATASET_DIR, self.__class__.__name__, 'uncompressed')
        self.t1_file_name, self.fl_file_name, self.annot_file_name = self.cfg.PROJECT.DATA_FILE_NAMES

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
    def get_data_tensor(t1_nii, fl_nii, annot_nii):
        t1_tensor, fl_tensor, annot_tensor = (
            torch.tensor(
                data=sitk.GetArrayFromImage(t1_nii),
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            ),
            torch.tensor(
                data=sitk.GetArrayFromImage(fl_nii),
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            ),
            torch.tensor(
                data=sitk.GetArrayFromImage(annot_nii),
                dtype=torch.long,
                device='cpu',
                requires_grad=False
            )
        )

        assert annot_tensor.unique(sorted=True).tolist() == [0, 1, 2], 'only expect labels of (0, 1, 2) in annotations'
        annot_tensor[annot_tensor == 2] = 255  # TODO check this with Unet3D to see what is done there.

        return t1_tensor, fl_tensor, annot_tensor

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        t1_path, fl_path, annot_path = self.find_data_files_path(sample_path)
        t1_nii, fl_nii, annot_nii = self.load_data(t1_path, fl_path, annot_path)
        meta_data = self.extract_data_meta(t1_nii, fl_nii, annot_nii)

        meta_data = self.run_sanity_checks(*meta_data)

        t1_tensor, fl_tensor, annot_tensor = self.get_data_tensor(t1_nii, fl_nii, annot_nii)

        image_tensor = torch.stack((t1_tensor, fl_tensor), dim=-1)  # D x H x W x C
        annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, annot_tensor = self.transform((image_tensor, annot_tensor, meta_data))

        return image_tensor, annot_tensor

    def __len__(self):
        return len(self.sample_path_list)
