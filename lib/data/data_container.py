#!/usr/bin/env python3

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from torch.utils.data import random_split
from .build import build_dataset
from data.WMHSegChal import collate_fn


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
        transformations_init = [
            tf.ToTensorImageVolume(),
            tf.OrientationToRAI(),
            tf.ResampleTo1mm(),
        ]
        if self.mode == 'train':
            transformations = [
                # tf.ResizeImageVolume(scale_factor=0.75),
                tf.RandomFlipImageVolume(p=0.5, dim=2)  # TODO later randomize dim with dim=-1
            ]
        elif self.mode in ('valid', 'test'):
            transformations = []
        else:
            raise NotImplementedError

        return torch_tf.Compose(
            transformations_init + transformations
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
