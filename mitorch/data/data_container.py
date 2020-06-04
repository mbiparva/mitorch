#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from torch.utils.data import random_split
from .build import build_dataset
from data.VolSet import collate_fn


# noinspection PyUnresolvedReferences
def ds_worker_init_fn(worker_id):
    # set numpy seed number for each worker separately
    assert torch.utils.data.get_worker_info().id == worker_id
    seed = torch.utils.data.get_worker_info().seed
    # needed for numpy random seed to be between 0 < seed < 2**32 - 1
    seed = seed if seed < 2**32 else seed % 2**32
    assert 0 < seed < 2 ** 32
    np.random.seed(seed)


# noinspection PyTypeChecker
class DataContainer:
    def __init__(self, mode, cfg):
        self.cfg = cfg
        self.dataset, self.dataloader = None, None
        self.mode = mode

        self.dataset_name, self.dl_params = self.init_dl_params()

        self.create_dataset()

        self.create_dataloader()

    def init_dl_params(self):
        if self.mode == 'train':
            dataset_name = self.cfg.TRAIN.DATASET
            batch_size = self.cfg.TRAIN.BATCH_SIZE
            shuffle = self.cfg.TRAIN.SHUFFLE
            drop_last = True
        elif self.mode == 'valid':
            dataset_name = self.cfg.TRAIN.DATASET
            batch_size = self.cfg.VALID.BATCH_SIZE
            shuffle = False
            drop_last = False
        elif self.mode == 'test':
            dataset_name = self.cfg.TEST.DATASET
            batch_size = self.cfg.TEST.BATCH_SIZE
            shuffle = False
            drop_last = False
        else:
            raise NotImplementedError
        return dataset_name, {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'drop_last': drop_last,
        }

    def create_transform(self):
        # --- HEAD ---
        transformations_head = [
            tf.ToTensorImageVolume(),
            tf.RandomOrientationTo('RPI'),
            # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
            tf.RandomResampleTomm(target_spacing=(1, 1, 1), target_spacing_scale=(0.2, 0.2, 0.2), prand=True),
        ]

        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
                # tf.CenterCropImageVolume(self.cfg.DATA.CROP_SIZE),
                # tf.RandomCropImageVolume(self.cfg.DATA.CROP_SIZE),
                tf.RandomResizedCropImageVolume(self.cfg.DATA.CROP_SIZE,
                                                scale=self.cfg.DATA.CROP_SCALE,
                                                uni_scale=self.cfg.DATA.UNI_SCALE),
                tf.RandomFlipImageVolume(dim=-1),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=False),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
            tf.NormalizeMeanStdVolume(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            ),
        ]

        return torch_tf.Compose(
            transformations_head + transformations_body + transformations_tail
        )

    def create_transform_hpo(self):
        transformations_head = [
            tf.ToTensorImageVolume(),
            (
                tf.RandomOrientationTo('RPI'),
                tf.RandomOrientationTo('RPI', prand=True)
            )[self.cfg.DATA.EXP.HEAD_ORI],
            (
                tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                tf.RandomResampleTomm(target_spacing=(1, 1, 1), target_spacing_scale=(0.2, 0.2, 0.2), prand=True),
            )[self.cfg.DATA.EXP.HEAD_RES],
        ]
        transformations_tail = (
            [
                tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
                tf.NormalizeMeanStdVolume(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
            ],
            [
                tf.NormalizeMeanStdVolume(
                    mean=[-0.06902332603931427, -0.0901104062795639],
                    std=[0.07958264648914337, 0.07952401041984558],
                    inplace=True
                ),
            ]
        )[self.cfg.DATA.EXP.TAIL]

        if self.mode == 'train':
            transformations_body = [
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=False),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
                (
                    tf.CenterCropImageVolume(self.cfg.DATA.CROP_SIZE),
                    tf.RandomCropImageVolume(self.cfg.DATA.CROP_SIZE),
                    tf.RandomResizedCropImageVolume(self.cfg.DATA.CROP_SIZE, scale=self.cfg.DATA.CROP_SCALE),
                )[self.cfg.DATA.EXP.BODY_CRO],
            ] + (
                [tf.RandomFlipImageVolume(dim=-1)],
                []
            )[self.cfg.DATA.EXP.BODY_FLI]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=False),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
            ]
        else:
            raise NotImplementedError

        return torch_tf.Compose(
            transformations_head + transformations_body + transformations_tail
        )

    def data_split(self):
        torch.manual_seed(self.cfg.RNG_SEED)
        n_tst = int(len(self.dataset) * self.cfg.PROJECT.TSR)
        n_traval = len(self.dataset) - n_tst
        n_tra = int(n_traval * self.cfg.PROJECT.TVSR)
        n_val = n_traval - n_tra
        tra, val, tst = random_split(
            self.dataset,
            (
                n_tra,
                n_val,
                n_tst,
            )
        )
        if self.mode == 'train':
            self.dataset = tra
        elif self.mode == 'valid':
            self.dataset = val
        elif self.mode == 'test':
            self.dataset = tst

    def create_dataset(self):
        transformations = self.create_transform()

        self.dataset = build_dataset(self.dataset_name, self.cfg, self.mode, transformations)

        self.data_split()

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
                                     worker_init_fn=ds_worker_init_fn,
                                     collate_fn=collate_fn,
                                     ** self.dl_params
                                     )
