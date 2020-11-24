#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import data.transforms_mitorch as tf
from torch.utils.data import DataLoader
import torchvision.transforms as torch_tf
from data.data_container import ds_worker_init_fn
from data.VolSet import collate_fn
from data.TestSetExt import TestSet
from models.build import build_model
import utils.checkpoint as checkops
from data.build import build_dataset
from utils.metrics import dice_coefficient_metric, jaccard_index_metric, hausdorff_distance_metric
from config.defaults import init_cfg
from netwrapper.net_wrapper import NetWrapperHFB, NetWrapperWMH
from test_net_single import test as test_single
from data.build_test_pipeline import build_transformations
import copy

KNOWN_TRANSFORMATIONS = (
    'noise',
    'contrast',
)
KNOWN_T_KEYS = (
    't_name',
    't_params',
)


def sanity_check_exp(exp):
    exp_perm_len = None
    for t in exp:
        assert all(
            i in KNOWN_T_KEYS for i in t.keys()
        ), f'unknown keys are defined in the transformations {t}'

        for u, v in t['t_params'].items():
            if exp_perm_len is None:
                exp_perm_len = len(v)
                continue
            assert exp_perm_len == len(v), 'expect t_params have a fixed length, getting {} and {} for {}'.format(
                exp_perm_len, len(v), u
            )

    assert exp_perm_len > 0, 'length of experiment permutation must be > 0'

    return exp_perm_len


def create_transformations(cfg, exp):
    transformations_head = [
        tf.ToTensorImageVolume(),
        tf.RandomOrientationTo('RPI'),
        tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
    ]

    transformations_body = build_transformations(cfg, exp)

    transformations_tail = [
        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
    ]

    return torch_tf.Compose(
        transformations_head + transformations_body + transformations_tail
    )


def define_exp_current(exp, j):
    exp_current = copy.deepcopy(exp)

    exp_description = dict()
    for k, t in enumerate(exp_current):
        t_name_k = f't_name_{k}'
        exp_description[t_name_k] = t['t_name']
        for p_key in t['t_params'].keys():
            t['t_params'][p_key] = t['t_params'][p_key][j]
            exp_description[f'{t_name_k}_{p_key}'] = t['t_params'][p_key]

    return exp_current, exp_description


def test_single_exp(cfg, exp):
    exp_results = list()

    # sanity check for t_params
    exp_perm_len = sanity_check_exp(exp)

    # create transformations
    for j in enumerate(range(exp_perm_len)):
        exp_current, exp_description = define_exp_current(exp, j)

        transformations = create_transformations(cfg, exp_current)

        output_single = test_single(cfg, transformations=transformations)

        output_single.update(exp_description)

        exp_results.append(output_single)

    return exp_results


def test(cfg):
    exp_ls = cfg.TEST.ROBUST_EXP_LIST
    exp_ls_results = list()

    for i, exp in enumerate(exp_ls):
        output_results = test_single_exp(cfg, exp)

        output_df = pd.DataFrame(output_results)

        exp_ls_results.append(output_df)

        print(output_df)

    # TODO whatever you want with exp_ls_results, e.g. save it to disk, visualize in graphs etc. I just print it.
