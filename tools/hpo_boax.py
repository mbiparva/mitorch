#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from config.defaults import init_cfg
from pprint import PrettyPrinter
from ax.service.managed_loop import optimize
from utils.hpo import *
import train_net_hpo

pp = PrettyPrinter(indent=4)
EXP_SEL = (0, 1, 2, 3, 4)[4]

# Experiment #1
hp_set = [
    [
        {
            'name': 'SOLVER.BASE_LR',
            'type': 'choice',
            'values': [
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5
            ],
        },
        {
            'name': 'SOLVER.OPTIMIZING_METHOD',
            'type': 'choice',
            'values': [
                'sgd',
                'adadelta',
                'adagrad',
                'rmsprop',
                'adam',
            ],
        },
        {
            'name': 'SOLVER.MOMENTUM',
            'type': 'choice',
            'values': [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95
            ],
        },
        {
            'name': 'SOLVER.WEIGHT_DECAY',
            'type': 'choice',
            'values': [
                1e-2,
                1e-3,
                1e-4,
                1e-5,
            ],
        },
        {
            'name': 'SOLVER.NESTEROV',
            'type': 'choice',
            'values': [
                False,
                True,
            ],
        },
        {
            'name': 'MODEL.DROPOUT_RATE',
            'type': 'choice',
            'values': [
                0.10,
                0.30,
                0.50,
                0.70,
                0.90,
            ],
        },
    ],
    [
        {
            'name': 'DATA.EXP.HEAD_ORI',
            'type': 'choice',
            'values': [
                0, 1,
            ],
        },
        {
            'name': 'DATA.EXP.HEAD_RES',
            'type': 'choice',
            'values': [
                0, 1,
            ],
        },
        {
            'name': 'DATA.EXP.BODY_CRO',
            'type': 'choice',
            'values': [
                0, 1, 2, 3
            ],
        },
        {
            'name': 'DATA.EXP.BODY_FLI',
            'type': 'choice',
            'values': [
                0, 1
            ],
        },
    ],
    [
        {
            'name': 'SOLVER.BASE_LR',
            'type': 'choice',
            'values': [
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5
            ],
        },
        {
            'name': 'SOLVER.OPTIMIZING_METHOD',
            'type': 'choice',
            'values': [
                'sgd',
                'adadelta',
                'adagrad',
                'rmsprop',
                'adam',
            ],
        },
        {
            'name': 'SOLVER.MOMENTUM',
            'type': 'choice',
            'values': [
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95
            ],
        },
        {
            'name': 'SOLVER.WEIGHT_DECAY',
            'type': 'choice',
            'values': [
                1e-2,
                1e-3,
                1e-4,
                1e-5,
            ],
        },
        {
            'name': 'SOLVER.NESTEROV',
            'type': 'choice',
            'values': [
                False,
                True,
            ],
        },
        {
            'name': 'MODEL.DROPOUT_RATE',
            'type': 'choice',
            'values': [
                0.25,
                0.50,
                0.75,
            ],
        },
        {
            'name': 'DATA.EXP.HEAD_ORI',
            'type': 'choice',
            'values': [
                0, 1,
            ],
        },
        {
            'name': 'DATA.EXP.HEAD_RES',
            'type': 'choice',
            'values': [
                0, 1,
            ],
        },
        {
            'name': 'DATA.EXP.BODY_CRO',
            'type': 'choice',
            'values': [
                0, 1, 2, 3
            ],
        },
        {
            'name': 'DATA.EXP.BODY_FLI',
            'type': 'choice',
            'values': [
                0, 1
            ],
        },
        {
            'name': 'DATA.EXP.INTENSITY',
            'type': 'choice',
            'values': [
                False, True
            ],
        },
        {
            'name': 'DATA.EXP.INTENSITY_SEL',
            'type': 'choice',
            'values': [
                0, 1, 2, 3, 4, 5, 6, 7, 8,
            ],
        },
        {
            'name': 'DATA.MAX_SIDE_SIZE',
            'type': 'choice',
            'values': [
                176, 192, 208, 224, 240, 256,
            ],
        },
        {
            'name': 'DATA.CROP_SIZE_FACTOR',
            'type': 'choice',
            'values': [
                1, 2, 3, 4,
            ],
        }
    ],
    [
        {
            'name': 'DATA.MAX_SIDE_SIZE',
            'type': 'choice',
            'values': [
                192,
                208,
            ],
        },
        {
            'name': 'DATA.CROP_SIZE',
            'type': 'choice',
            'values': [
                144,
                160,
                176,
                192
            ],
        }
    ],
    [
        {
            'name': 'SOLVER.BASE_LR',
            'type': 'choice',
            'values': [
                1e-1, 1e-2, 1e-3, 1e-4, 1e-5
            ],
        },
        {
            'name': 'SOLVER.OPTIMIZING_METHOD',
            'type': 'choice',
            'values': [
                # 'sgd', 'adadelta', 'adagrad', 'rmsprop', 'adam',
                'sgd', 'adam',
            ],
        },
        {
            'name': 'SOLVER.MOMENTUM',
            'type': 'choice',
            'values': [
                0.5, 0.6, 0.7, 0.8, 0.9, 0.95
            ],
        },
        {
            'name': 'SOLVER.WEIGHT_DECAY',
            'type': 'choice',
            'values': [
                1e-2, 1e-3, 1e-4, 1e-5,
            ],
        },
        # {
        #     'name': 'SOLVER.NESTEROV',
        #     'type': 'choice',
        #     'values': [
        #         False, True,
        #     ],
        # },
        {
            'name': 'MODEL.DROPOUT_RATE',
            'type': 'choice',
            'values': [
                0.10, 0.30, 0.50, 0.70, 0.90,
            ],
        },
        {
            'name': 'DATA.EXP.BODY_FLI',
            'type': 'choice',
            'values': [
                0, 1, 2, 3, 4,
            ],
        },
        {
            'name': 'DATA.EXP.INTENSITY',
            'type': 'choice',
            'values': [
                False, True
            ],
        },
        {
            'name': 'DATA.EXP.INTENSITY_SEL',
            'type': 'choice',
            'values': [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            ],
        },
        {
            'name': 'DATA.EXP.TAIL_NORM',
            'type': 'choice',
            'values': [
                0, 1, 2,
            ],
        },
        {
            'name': 'NVT.SELECTION_LB',
            'type': 'choice',
            'values': [
                2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000
            ],
        },
        # {
        #     'name': 'MODEL.LOSSES[0]['name']',
        #     'type': 'choice',
        #     'values': [
        #         'CrossEntropyLoss', 'DiceLoss',  # 'FocalLoss', 'LovaszLoss'
        #     ],
        # }
    ],
][EXP_SEL]


# TODO find a better consistent solution later
# used only by BOAX since we cannot pass them directly
cfg_g, tb_hps_sw_g, len_exps_g, hpo_parent_dir_g = None, None, None, None


def boax_train_evaluate(parameterization):
    hps_dict, cfg = hp_gen_set_cfg(tuple(parameterization.items()), cfg_g)

    print('BoAx experimentation; len of experiments {}; parameters {}'.format(len_exps_g, hps_dict))

    cfg = init_cfg(cfg, parent_dir=hpo_parent_dir_g)
    # cfg.WMH.MAX_SIDE_SIZE = cfg.DATA.MAX_SIDE_SIZE - 16*2
    # cfg.DATA.CROP_SIZE = cfg.DATA.MAX_SIDE_SIZE - 16*cfg.DATA.CROP_SIZE_FACTOR
    eval_met_dict = train_net_hpo.hpo_train_eval_instance(cfg)

    hps_dict = {u: ', '.join(list(map(str, v))) if isinstance(v, (tuple, list)) else v for u, v in hps_dict.items()}
    tb_hps_sw_g.add_hparams(hps_dict, eval_met_dict)

    return eval_met_dict['hparam/{}_ep'.format(cfg.HPO.EVAL_METRIC)]


def set_global_vars(cfg, tb_hps_sw, len_exps, hpo_parent_dir):
    global cfg_g, tb_hps_sw_g, len_exps_g, hpo_parent_dir_g
    cfg_g, tb_hps_sw_g, len_exps_g, hpo_parent_dir_g = cfg, tb_hps_sw, len_exps, hpo_parent_dir


def run(cfg, tb_hps_sw, len_exps, hpo_parent_dir):
    set_global_vars(cfg, tb_hps_sw, len_exps, hpo_parent_dir)
    cfg.hp_set = hp_set

    best_parameters, values, experiment, model = optimize(
        parameters=hp_set,
        evaluation_function=boax_train_evaluate,
        objective_name=cfg.HPO.EVAL_METRIC,
        total_trials=cfg.HPO.TOTAL_TRIALS,
        minimize=True,
    )

    pp.pprint(best_parameters)
    pp.pprint(values)
    pp.pprint(experiment)
    pp.pprint(model)
