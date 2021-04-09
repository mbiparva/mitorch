#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
from epoch_loop import EpochLoop
from torch.utils.tensorboard import SummaryWriter
from pprint import PrettyPrinter
from utils.hpo import *
import hpo_manual
import hpo_boax

pp = PrettyPrinter(indent=4)


def hpo_train_eval_instance(cfg):
    epoch_loop = EpochLoop(cfg)

    epoch_loop.main()

    return {
        '{}/{}'.format('hparam', k): m_avg
        for k, m_avg in epoch_loop.evaluator.meters.get_avg_for_tb()
    }


def hpo_main(cfg):
    assert cfg.TRAIN.ENABLE
    assert not cfg.TEST.ENABLE

    len_exps = -1
    cfg.SOLVER.MAX_EPOCH = cfg.HPO.MAX_EPOCH

    hpo_parent_dir = '{}_hpo_{}'.format(cfg.MODEL.ID, cfg.HPO.MODE)
    hpo_output_dir = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, cfg.TRAIN.DATASET, hpo_parent_dir)
    tb_hps_sw = SummaryWriter(hpo_output_dir)
    os.rmdir(cfg.OUTPUT_DIR)  # it is useless, we use hpo_output_dir instead

    try:
        if cfg.HPO.MODE == 'MAN':
            len_exps = len_hp_set(hpo_manual.hp_set)
            hpo_manual.run(cfg, tb_hps_sw, len_exps, hpo_parent_dir)
        elif cfg.HPO.MODE == 'BOAX':
            len_exps = len_hp_param(hpo_boax.hp_set)
            hpo_boax.run(cfg, tb_hps_sw, len_exps, hpo_parent_dir)
        else:
            raise NotImplementedError
    except KeyboardInterrupt:
        print('*** The experimentation is terminated by a keyboard interruption')

    print('#num of trial cases:', len_exps)
