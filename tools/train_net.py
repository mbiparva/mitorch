#!/usr/bin/env python3

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
