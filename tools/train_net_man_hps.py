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

hp_set = {
    'solver_base_lr': (
        1e-2,
        1e-3,
        1e-4,
    ),
    'solver_optimizing_method': (
        'sgd',
        'adam',
    ),
    'data_padding_mode': (
        'mean',
        'median',
        'min',
        'max'
    ),
}
SOLVER_MAX_EPOCH = 2


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
    assert isinstance(in_item, str) and len(in_item)
    key, value = in_item
    key_list = key.split('.')
    if len(key_list) == 1:
        setattr(cfg, key_list[0], value)
    elif len(key_list) == 1:
        key_par = cfg.get(key_list[0], None)
        assert key_par is not None, 'key parent {} not found'.format(key_par)
        setattr(key_par, key_list[1], value)
        setattr(cfg, key_list[0], key_par)
    else:
        raise ValueError('only accept one- or two-level attribute hierarchy but got {}'.format(key))


def hp_gen(cfg):
    for hps in product(hp_set.values()):
        hps_dict = dict()
        for k, v in zip(hp_set.keys(), hps):
            set_hp_cfg(cfg, (k, v))
            hps_dict[k] = v

        yield hps_dict, cfg


def train(cfg):
    cfg.SOLVER.MAX_EPOCH = SOLVER_MAX_EPOCH

    tb_logger_dir = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, cfg.TRAIN.DATASET, cfg.MODEL.ID, 'man_hps')
    tb_hps_sw = SummaryWriter(tb_logger_dir)

    for hps_dict, cfg in hp_gen(cfg):
        eval_met_dict = train_hp(cfg)

        tb_hps_sw.add_hparams(hps_dict, eval_met_dict)

    tb_hps_sw.close()
