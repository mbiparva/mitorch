#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from epoch_loop import EpochLoop


def test(cfg):
    assert not cfg.TRAIN.ENABLE
    assert cfg.TEST.ENABLE
    cfg.VALID.ENABLE = True

    epoch_loop = EpochLoop(cfg)

    try:
        epoch_loop.main()
    except KeyboardInterrupt:
        print('*** The experiment is terminated by a keyboard interruption')


if __name__ == '__main__':
    raise NotImplementedError('Please use run_net.py for now')
