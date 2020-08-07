#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)


from config.defaults import init_cfg
from utils.hpo import *
import train_net_hpo

EXP_SEL = 7

hp_set = {
    0: {
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
    1: {
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
    },
    2: {
        'TRAIN.BATCH_SIZE': (
            1, 2, 4
        ),
    },
    3: {
        'DATA.MAX_SIDE_SIZE': (
            192, 208, 224, 240, 256,
        ),
    },
    4: {
        'DATA.CROP_SIZE': (
            3, 2, 1, 0,
        ),
        'DATA.CROP_SCALE': (
            (0.7, 1.0),
            (0.7, 0.9),
            (0.9, 1.0),
            (0.8, 1.0),
            (0.8, 0.9),

        )
    },
    5: {
        'MODEL.WHL_NUM_DEPTH_SHEETS': (
            2, 4, 8,
        ),
        'MODEL.WHL_SEG_THR': (
            0.12, 0.25, 0.5,
        )
    },
    6: {
        'DATA.EXP.INTENSITY_SEL': (
            0, 1, 2, 3, 4, 5, 6, 7, 8
        ),
    },
    7: {
        'NVT.RANDOM_CROP_THRESHOLD': (
            0, 32, 64, 128, 256, 512, 1024
        ),
        # 'NVT.NUM_MULTI_PATCHES': (
        #     4, 8, 16, 32, 64
        # ),
        # 'NVT.SELECTION_LB': (
        #     0, 100, 200, 400, 800
        #     16000, 32000, 64000, 128000, 256000, 512000
        # ),
    },
}[EXP_SEL]


def run(cfg, tb_hps_sw, len_exps, hpo_parent_dir):
    cfg.hp_set = list(hp_set.items())
    r_s, r_e = exp_range_finder(cfg, len_exps)

    for i, (hps_dict, cfg) in enumerate(hp_gen(cfg, hp_set)):
        if not (r_s <= i < r_e):
            continue
        print('manual hps iter {:02}|{} started: {}'.format(i, len_exps-1, hps_dict))
        cfg = init_cfg(cfg, parent_dir=hpo_parent_dir)
        # cfg.WMH.MAX_SIDE_SIZE = cfg.DATA.MAX_SIDE_SIZE - 16*2
        # cfg.DATA.CROP_SIZE = cfg.DATA.MAX_SIDE_SIZE - 16 * cfg.DATA.CROP_SIZE_FACTOR
        eval_met_dict = train_net_hpo.hpo_train_eval_instance(cfg)

        hps_dict = {u: ', '.join(list(map(str, v))) if isinstance(v, (tuple, list)) else v for u, v in hps_dict.items()}
        tb_hps_sw.add_hparams(hps_dict, eval_met_dict)

    tb_hps_sw.close()
