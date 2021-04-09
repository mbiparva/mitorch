#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from epoch_loop import EpochLoop


def train(cfg):
    assert cfg.TRAIN.ENABLE
    assert not cfg.TEST.ENABLE

    epoch_loop = EpochLoop(cfg)

    try:
        epoch_loop.main()
    except KeyboardInterrupt:
        print('*** The experiment is terminated by a keyboard interruption')

    # TODO: setup experimentation platform e.g. Manually or AX/BO TORCH


if __name__ == '__main__':
    raise NotImplementedError('Please use run_net.py for now')
