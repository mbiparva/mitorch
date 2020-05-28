#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)


from config.defaults import init_cfg
from utils.hpo import *
import train_net_hpo

EXP_SEL = (0, 1)[0]

hp_set = [
    {
        'SOLVER.BASE_LR': (
            1e-2,
            1e-3,
            1e-4,
        ),
        'SOLVER.OPTIMIZING_METHOD': (
            'sgd',
            'adadelta',
            'adagrad',
            'rmsprop',
            'adam',
        ),
        'SOLVER.MOMENTUM': (
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95
        ),
        'SOLVER.WEIGHT_DECAY': (
            1e-2,
            1e-3,
            1e-4,
            1e-5,
        ),
        'SOLVER.NESTEROV': (
            False,
            True,
        )
    },
    {
        'DATA.EXP.HEAD_ORI': (
            0, 1
        ),
        'DATA.EXP.HEAD_RES': (
            0, 1
        ),
        'DATA.EXP.TAIL': (
            0, 1
        ),
        'DATA.EXP.BODY_CRO': (
            0, 1, 2
        ),
        'DATA.EXP.BODY_FLI': (
            0, 1
        ),
    }
][EXP_SEL]


def run(cfg, tb_hps_sw, len_exps):
    r_s, r_e = exp_range_finder(cfg, len_exps)

    for i, (hps_dict, cfg) in enumerate(hp_gen(cfg, hp_set)):
        if not (r_s <= i < r_e):
            continue
        print('manual hps iter {:02}|{} started: {}'.format(i, len_exps-1, hps_dict))
        cfg = init_cfg(cfg)
        eval_met_dict = train_net_hpo.hpo_train_eval_instance(cfg)

        tb_hps_sw.add_hparams(hps_dict, eval_met_dict)

    tb_hps_sw.close()
