#!/usr/bin/env python3

import os
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
# import data.spatial_transformation as tf
import torchvision.transforms as tf
from torch.utils.data import random_split
from .build import build_dataset


def ds_worker_init_fn(worker_id):
    # np.random.seed(datetime.datetime.now().microsecond + worker_id)
    assert torch.utils.data.get_worker_info().id == worker_id
    seed = torch.utils.data.get_worker_info().seed
    # needed for numpy random seed to be between 0 < seed < 2**32 - 1
    seed = seed if seed < 2**32 else seed % 2**32
    assert 0 < seed < 2 ** 32
    np.random.seed(seed)


def data_collator():
    raise NotImplementedError()


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
            batch_size = self.cfg.TRAIN.BATCH_SIZE
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
        # h, w = self.cfg.SPATIAL_INPUT_SIZE
        # assert h == w
        h = 32
        transformations_final = [
            tf.ToTensor(),
            # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet-driven
        ]
        if self.mode == 'train':
            transformations = [
                # Transformation.RandomCornerCrop(240, crop_scale=(240, 224, 192, 168), border=0.25),
                # Transformation.RandomCornerCrop(240, crop_scale=(0.66, 1.0), border=0.25),
                tf.Resize(h),
                tf.CenterCrop(h),
                tf.RandomHorizontalFlip()
            ]
        elif self.mode in ('valid', 'test'):
            transformations = [
                tf.Resize(h),
                tf.CenterCrop(h),
            ]
        else:
            raise NotImplementedError

        return tf.Compose(
            transformations + transformations_final
        )

    def data_split(self):
        torch.manual_seed(110)
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
        # TODO : Either let the dataset take care of the transformation or do it like so
        spatial_transform = self.create_transform()

        # Construct the dataset
        self.dataset = build_dataset(self.dataset_name, self.cfg, self.mode, spatial_transform)

        # TODO: Either let the dataset to take care of split, or do it right here
        self.data_split()

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     # batch_size=self.batch_size,
                                     # shuffle=self.shuffle,
                                     num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
                                     # drop_last=True,
                                     worker_init_fn=ds_worker_init_fn,
                                     collate_fn=None,  # data_collator,
                                     ** self.dl_params
                                     )
