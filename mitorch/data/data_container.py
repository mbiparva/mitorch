#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from torch.utils.data import random_split
from .build import build_dataset
from .build_transformations import build_transformations
from data.VolSet import collate_fn as collate_fn_vol
from data.NeuroSegSets import collate_fn as collate_fn_pat
import os


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
        self.dataset, self.dataloader, self.sampler = None, None, None
        self.mode = mode

        self.dataset_name, self.dl_params = self.init_dl_params()

        self.create_dataset()

        self.create_dataloader()

    def init_dl_params(self):
        collate_fn = collate_fn_pat if self.cfg.NVT.ENABLE else collate_fn_vol

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
            'collate_fn': collate_fn,
        }

    def create_transform_single(self):
        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                # tf.ToTensorImageVolume(),
                # tf.RandomOrientationTo('RPI'),
                # tf.RandomOrientationTo('RPI', prand=True),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1), target_spacing_scale=(0.2, 0.2, 0.2), prand=True),
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),

                # tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                # tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
                # tf.CenterCropImageVolume(self.cfg.DATA.CROP_SIZE),
                # tf.RandomCropImageVolume(self.cfg.DATA.CROP_SIZE),
                # tf.RandomResizedCropImageVolume(self.cfg.DATA.CROP_SIZE,
                #                                 scale=self.cfg.DATA.CROP_SCALE,
                #                                 uni_scale=self.cfg.DATA.UNI_SCALE),
                # tf.RandomFlipImageVolume(dim=-1),

                # tf.RandomBrightness(value=0.1, prand=True, channel_wise=True),
                # tf.RandomContrast(value=0.1, prand=True, channel_wise=True),
                # tf.RandomGamma(value=0.1, prand=True, channel_wise=True),
                # tf.LogCorrection(inverse=(False, True)[1], channel_wise=True),
                # tf.SigmoidCorrection(inverse=(False, True)[1], channel_wise=True),
                # tf.HistEqual(num_bins=256, channel_wise=True),
                # tf.AdditiveNoise(sigma=0.1, noise_type=('gaussian', 'rician', 'rayleigh')[0], randomize_type=False,
                #                  out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                # tf.ToTensorImageVolume(),
                # tf.RandomOrientationTo('RPI'),
                # tf.RandomOrientationTo('RPI', prand=True),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),

                # tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                # tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),

                # tf.HistEqual(num_bins=256, channel_wise=True),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            # tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
            # tf.NormalizeMeanStdVolume(
            #     mean=self.cfg.DATA.MEAN,
            #     std=self.cfg.DATA.STD,
            #     inplace=True
            # ),
        ]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )

    def create_transform_hpo_brain(self):
        if self.mode == 'train':
            transformations_body = [
                tf.ToTensorImageVolume(),
                (
                    tf.RandomOrientationTo('RPI'),
                    tf.RandomOrientationTo('RPI', prand=True)
                )[self.cfg.DATA.EXP.HEAD_ORI],
                (
                    tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                    tf.RandomResampleTomm(target_spacing=(1, 1, 1), target_spacing_scale=(0.2, 0.2, 0.2), prand=True),
                )[self.cfg.DATA.EXP.HEAD_RES],

                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=False),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
            ] + (
                    [],
                    [tf.CenterCropImageVolume(self.cfg.DATA.CROP_SIZE)],
                    [tf.RandomCropImageVolume(self.cfg.DATA.CROP_SIZE)],
                    [tf.RandomResizedCropImageVolume(self.cfg.DATA.CROP_SIZE, scale=self.cfg.DATA.CROP_SCALE)],
            )[self.cfg.DATA.EXP.BODY_CRO] + (
                [],
                [tf.RandomFlipImageVolume(dim=-1)],
            )[self.cfg.DATA.EXP.BODY_FLI] + (
                [],
                [[
                    tf.RandomBrightness(value=0.25, prand=True, channel_wise=True),
                    tf.RandomContrast(value=0.25, prand=True, channel_wise=True),
                    tf.RandomGamma(value=2.0, prand=True, channel_wise=True),
                    tf.LogCorrection(inverse=(False, True)[0], channel_wise=True),
                    tf.SigmoidCorrection(inverse=(False, True)[0], channel_wise=True),
                    tf.HistEqual(num_bins=512, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.5, noise_type=('gaussian', 'rician', 'rayleigh')[0], randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.5, noise_type=('gaussian', 'rician', 'rayleigh')[1], randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.5, noise_type=('gaussian', 'rician', 'rayleigh')[2], randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                ][self.cfg.DATA.EXP.INTENSITY_SEL]]
            )[self.cfg.DATA.EXP.INTENSITY]

        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                tf.RandomResampleTomm(target_spacing=(1, 1, 1)),

                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                # tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),

                # tf.HistEqual(num_bins=256, channel_wise=True),
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
            transformations_body + transformations_tail
        )

    def create_transform_hpo(self):
        if self.mode == 'train':
            transformations_body = [
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),
            ] + (
                [],
                [tf.RandomFlipImageVolume(dim=-1)],
                [tf.RandomFlipImageVolume(dim=0)],
                [tf.RandomFlipImageVolume(dim=1)],
                [tf.RandomFlipImageVolume(dim=2)],
            )[self.cfg.DATA.EXP.BODY_FLI] + (
                [],
                [[
                    tf.RandomBrightness(value=0.1, prand=True, channel_wise=True),
                    tf.RandomContrast(value=0.1, prand=True, channel_wise=True),
                    tf.RandomGamma(value=0.1, prand=True, channel_wise=True),
                    tf.LogCorrection(inverse=False, channel_wise=True),
                    tf.LogCorrection(inverse=True, channel_wise=True),
                    tf.SigmoidCorrection(inverse=False, channel_wise=True),
                    tf.SigmoidCorrection(inverse=True, channel_wise=True),
                    tf.HistEqual(num_bins=128, channel_wise=True),
                    tf.HistEqual(num_bins=256, channel_wise=True),
                    tf.HistEqual(num_bins=512, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.1, noise_type='gaussian', randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.1, noise_type='rician', randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                    tf.AdditiveNoise(sigma=0.1, noise_type='rayleigh', randomize_type=False,
                                     out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
                ][self.cfg.DATA.EXP.INTENSITY_SEL]]
            )[self.cfg.DATA.EXP.INTENSITY]

        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),
            ]

        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = (
            [],
            [
                tf.NormalizeMeanStdVolume(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
            ],
            [
                tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
                tf.NormalizeMeanStdVolume(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
            ]
        )[self.cfg.DATA.EXP.TAIL_NORM]
        if self.cfg.DATA.EXP.TAIL_NORM == 1:
            self.cfg.DATA.MEAN: [256.42889404296875, 380.6856689453125]  # without MinMax
            self.cfg.DATA.STD: [64.1461410522461, 78.29484558105469]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )

    def create_transform(self):
        CHOOSE_BEST_TRANSFORMS = (False, True)[1]
        CHOOSE_HPO_TRANSFORMS = (False, True)[0]
        if CHOOSE_BEST_TRANSFORMS:
            transformations = build_transformations(self.dataset_name, self.cfg, self.mode)()
        else:
            if CHOOSE_HPO_TRANSFORMS:
                transformations = self.create_transform_hpo()
            else:
                transformations = self.create_transform_single()

        self.save_transformations_str(transformations)

        return transformations

    def data_split_pa_ind(self):
        with open(os.path.join(self.cfg.PROJECT.DATASET_DIR, 'wmh_validation_subjs.txt'), 'r') as fh:
            ind_list = fh.readlines()
        ind_list = [i.strip() for i in ind_list]
        ind_list_index = list()
        non_ind_list_index = list()
        for i, s in enumerate(self.dataset.sample_path_list):
            s_name = s.rpartition('/')[-1]
            if s_name in ind_list:
                ind_list_index.append(i)
            else:
                non_ind_list_index.append(i)
        if self.mode == 'train':
            self.dataset = torch.utils.data.Subset(self.dataset, non_ind_list_index)
        elif self.mode == 'valid':
            self.dataset = torch.utils.data.Subset(self.dataset, ind_list_index)
        elif self.mode == 'test':
            raise NotImplementedError('undefined in this function')

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

        if self.cfg.TRAIN and self.cfg.TRAIN.DATASET == 'SRIBIL' and self.cfg.PROJECT.PA_INDICES:
            self.data_split_pa_ind()
        else:
            self.data_split()

        if self.mode == 'train' and self.cfg.NVT.ENABLE and self.cfg.NVT.REPEAT_DATASET > 1:
            self.dataset = ConcatDataset([self.dataset]*self.cfg.NVT.REPEAT_DATASET)

        if self.cfg.DDP:
            try:  # torch 1.5.0 on mist has issue with seed, remove it later
                self.sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset,
                    num_replicas=self.cfg.DDP_CFG.WORLD_SIZE,
                    rank=self.cfg.DDP_CFG.RANK,
                    shuffle=self.dl_params['shuffle'],
                    seed=self.cfg.RNG_SEED,
                )
            except Exception:
                self.sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset,
                    num_replicas=self.cfg.DDP_CFG.WORLD_SIZE,
                    rank=self.cfg.DDP_CFG.RANK,
                    shuffle=self.dl_params['shuffle'],
                )
            finally:
                self.dl_params['shuffle'] = False

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     sampler=self.sampler,
                                     num_workers=self.cfg.DATA_LOADER.NUM_WORKERS,
                                     pin_memory=self.cfg.DATA_LOADER.PIN_MEMORY,
                                     worker_init_fn=ds_worker_init_fn,
                                     ** self.dl_params
                                     )

    def save_transformations_str(self, transformations):
        self.cfg.__setitem__(f'transformations_{self.mode}'.upper(), transformations.__str__())
