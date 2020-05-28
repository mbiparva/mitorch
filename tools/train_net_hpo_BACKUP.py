#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import os
from epoch_loop import EpochLoop
from torch.utils.tensorboard import SummaryWriter
from config.defaults import init_cfg
from pprint import PrettyPrinter
from ax.service.managed_loop import optimize
from utils.hpo import *

pp = PrettyPrinter(indent=4)

HPO_MODE = ('MAN', 'BOAX')[0]
SOLVER_MAX_EPOCH = 5

# STAGE #1
hp_set_manual = {
    'SOLVER.BASE_LR': (
        1e-2,
        1e-3,
        1e-4,
    ),
    'SOLVER.OPTIMIZING_METHOD': (
        'sgd',
        'adam',
    ),
    'MODEL.LOSS_FUNC': (
        'L1Loss',
        'MSELoss',
        'SmoothL1Loss',
    )
}

# STAGE #1
# hp_set_manual = {
#     'DATA.EXP.HEAD_ORI': (0, 1),
#     'DATA.EXP.HEAD_RES': (0, 1),
#     'DATA.EXP.TAIL': (0, 1),
#     'DATA.EXP.BODY_CRO': (0, 1, 2),
#     'DATA.EXP.BODY_FLI': (0, 1),
# }


hp_set_boax = [
    {
        'name': 'SOLVER.BASE_LR',
        'type': 'choice',
        'values': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, ),
    },
    {
        'name': 'SOLVER.OPTIMIZING_METHOD',
        'type': 'choice',
        'values': ('sgd', 'adam', )
    },
    {
        'name': 'MODEL.LOSS_FUNC',
        'type': 'choice',
        'values': ('L1Loss', 'MSELoss', 'SmoothL1Loss', )
    },
]
# TODO find a better consistent solution later
# used only by BOAX since we cannot pass them directly
cfg_g, tb_hps_sw_g, len_exps_g = None, None, None


def train_hp(cfg):
    epoch_loop = EpochLoop(cfg)

    epoch_loop.main()

    return {
        '{}/{}'.format('hparam', k): m_avg
        for k, m_avg in epoch_loop.evaluator.meters.get_avg_for_tb()
    }


def hpo_manual(cfg, tb_hps_sw, len_exps):
    r_s, r_e = exp_range_finder(cfg, len_exps)

    for i, (hps_dict, cfg) in enumerate(hp_gen(cfg, hp_set_manual)):
        if not (r_s <= i < r_e):
            continue
        print('manual hps iter {:02}|{} started: {}'.format(i, len_exps-1, hps_dict))
        cfg = init_cfg(cfg)
        eval_met_dict = train_hp(cfg)

        tb_hps_sw.add_hparams(hps_dict, eval_met_dict)

    tb_hps_sw.close()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def boax_train_evaluate(parameterization):
    global cfg_g, tb_hps_sw_g, len_exps_g

    hps_dict, cfg = hp_gen_set_cfg(tuple(parameterization.items()), cfg_g)

    print('BoAx experimentation; len of experiments {}; parameters {}'.format(len_exps_g, hps_dict))

    cfg = init_cfg(cfg)
    eval_met_dict = train_hp(cfg)

    tb_hps_sw_g.add_hparams(hps_dict, eval_met_dict)

    return eval_met_dict['hparam/{}_ep'.format(cfg.HPO.EVAL_METRIC)]


def set_global_vars(cfg, tb_hps_sw, len_exps):
    global cfg_g, tb_hps_sw_g, len_exps_g
    cfg_g, tb_hps_sw_g, len_exps_g = cfg, tb_hps_sw, len_exps


def hpo_boax(cfg, tb_hps_sw, len_exps):
    set_global_vars(cfg, tb_hps_sw, len_exps)

    best_parameters, values, experiment, model = optimize(
        parameters=hp_set_boax,
        evaluation_function=boax_train_evaluate,
        objective_name='accuracy',
    )

    pp.pprint(best_parameters)
    pp.pprint(values)
    pp.pprint(experiment)
    pp.pprint(model)


def train_hpo(cfg):
    assert cfg.TRAIN.ENABLE
    assert not cfg.TEST.ENABLE

    cfg.SOLVER.MAX_EPOCH = SOLVER_MAX_EPOCH

    tb_logger_dir = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, cfg.TRAIN.DATASET, cfg.MODEL.ID + '_man_hps')
    tb_hps_sw = SummaryWriter(tb_logger_dir)

    try:
        if HPO_MODE == 'MAN':
            len_exps = len_hp_set(hp_set_manual)
            hpo_manual(cfg, tb_hps_sw, len_exps)
        elif HPO_MODE == 'BOAX':
            len_exps = len_hp_param(hp_set_boax)
            hpo_boax(cfg, tb_hps_sw, len_exps)
    except KeyboardInterrupt:
        print('*** The experimentation is terminated by a keyboard interruption')
