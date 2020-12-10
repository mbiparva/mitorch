#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import pandas as pd
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from test_net_single import test as test_single
from data.build_test_pipeline import build_transformations
import copy
import logging
from datetime import datetime
import pprint


KNOWN_TRANSFORMATIONS = (
    'noise',
    'contrast',
    'rotate',
    'shear',
    'translate',
    'scale',
    'spike',
    'ghosting',
    'blur',
    'biasfield',
    'swap',
    'motion',
)
KNOWN_T_KEYS = (
    't_name',
    't_params',
)


def setup_logger():
    local_logger = logging.getLogger(__name__)
    local_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler if you wish
    file_handler = logging.FileHandler('/tmp/test_error_output_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    # create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    local_logger.addHandler(file_handler)
    local_logger.addHandler(stream_handler)

    return local_logger


logger = setup_logger()


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
    for j in range(exp_perm_len):
        logger.info(f'experiment permutation: {j}|{exp_perm_len}')

        exp_current, exp_description = define_exp_current(exp, j)

        transformations = create_transformations(cfg, exp_current)

        output_single = test_single(cfg, transformations=transformations, save_pred_flag=True, eval_pred_flag=True)

        output_single.update(exp_description)

        exp_results.append(output_single)

        logger.info(f'{"".join(["-"*20])}\n')

    return exp_results


def process_output_results(exp_ls_results):
    # TODO whatever you want with exp_ls_results, e.g. save it to disk, visualize in graphs etc. I just print it.
    for i, exp, output_df in exp_ls_results:
        logger.info(f'{i} --- \n{pprint.pformat(exp)}:\n{output_df}\n\n')


def test(cfg):
    exp_ls = cfg.TEST.ROBUST_EXP_LIST
    exp_ls_results = list()

    for i, exp in enumerate(exp_ls):
        logger.info(f'started testing experiment --- \n{i:03d}:{pprint.pformat(exp)}')

        output_results = test_single_exp(cfg, exp)

        output_df = pd.DataFrame(output_results)

        exp_ls_results.append((i, exp, output_df))

        logger.info(f'experiment {i} is done --- \n{output_df}\n')

    process_output_results(exp_ls_results)