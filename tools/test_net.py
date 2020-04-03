#!/usr/bin/env python3

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
