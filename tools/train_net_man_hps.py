#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import os
from epoch_loop import EpochLoop
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from config.defaults import init_cfg

SOLVER_MAX_EPOCH = 5
hp_set = {
    'SOLVER.BASE_LR': (
        1e-2,
        1e-3,
        1e-4,
    ),
    'SOLVER.OPTIMIZING_METHOD': (
        'sgd',
        'adam',
    ),
    'DATA.PADDING_MODE': (
        'mean',
        'median',
        'min',
        'max'
    ),
}


def train_hp(cfg):
    assert cfg.TRAIN.ENABLE
    assert not cfg.TEST.ENABLE

    epoch_loop = EpochLoop(cfg)

    try:
        epoch_loop.main()
    except KeyboardInterrupt:
        print('*** The experiment is terminated by a keyboard interruption')

    return {
        '{}/{}'.format('hparam', k): m_avg
        for k, m_avg in epoch_loop.evaluator.meters.get_avg_for_tb()
    }


def set_hp_cfg(cfg, in_item):
    key, value = in_item
    assert isinstance(key, str) and len(key)
    key_list = key.split('.')
    key_par = cfg
    for i, k in enumerate(key_list):
        if i == len(key_list) - 1:
            break
        key_par = cfg.get(k, None)
    setattr(key_par, key_list[1], value)

    return cfg


def hp_gen(cfg):
    for hps in product(*hp_set.values()):
        hps_dict = dict()
        for k, v in zip(hp_set.keys(), hps):
            cfg = set_hp_cfg(cfg, (k, v))
            hps_dict[k] = v

        yield hps_dict, cfg


def train(cfg):
    cfg.SOLVER.MAX_EPOCH = SOLVER_MAX_EPOCH

    tb_logger_dir = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, cfg.TRAIN.DATASET, cfg.MODEL.ID+'_man_hps')
    tb_hps_sw = SummaryWriter(tb_logger_dir)

    for i, (hps_dict, cfg) in enumerate(hp_gen(cfg)):
        print('manual hps iter {:02} started: {}'.format(i, hps_dict))
        cfg = init_cfg(cfg)
        eval_met_dict = train_hp(cfg)

        tb_hps_sw.add_hparams(hps_dict, eval_met_dict)

    tb_hps_sw.close()
