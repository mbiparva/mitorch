#!/usr/bin/env python3

"""Meters."""

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)


import datetime
import numpy as np
from collections import defaultdict, deque
from fvcore.common.timer import Timer

import utils.logging as logging
import utils.misc as misc

logger = logging.get_logger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = (0,)*4
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TVTMeter(object):
    """
    Measure training, validation, and testing stats.
    """

    def __init__(self, epoch_iters, cfg, meter_names):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        assert isinstance(meter_names, (tuple, list)) and len(meter_names) > 0
        assert all([isinstance(i, str) for i in meter_names])
        assert 'loss' in meter_names, 'loss is forgotten in the meter names'
        self.meter_names = meter_names
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.lr = None
        self.num_samples = 0

        self.create_meters()

    def meter_setter(self, m, t, v):
        """
        m (str): meter name
        t (str): meter type (mb or ep)
        v (object): any value
        """
        return setattr(self, '{}_{}'.format(m, t), v)

    def meter_getter(self, m, t):
        """
        m (str): meter name
        t (str): meter type (mb or ep)
        """
        return getattr(self, '{}_{}'.format(m, t))

    def create_meters(self):
        for m in self.meter_names:
            # Current minibatch errors (smoothed over a window).
            self.meter_setter(m, 'mb', ScalarMeter(self._cfg.LOG_PERIOD))
            # Epoch stats
            self.meter_setter(m, 'ep', AverageMeter())

    def reset(self):
        """
        Reset the Meter.
        """
        self.lr = None
        self.num_samples = 0
        for m in self.meter_names:
            self.meter_getter(m, 'mb').reset()
            self.meter_getter(m, 'ep').reset()

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, lr, mb_size, **kwargs):
        """
        Update the current stats.
        Args:
            lr (float): learning rate.
            mb_size (int): mini batch size.
            **kwargs (dict): dictionary of meters and values
        """
        assert len(self.meter_names) == len(kwargs), 'some meters are missing'
        assert set(self.meter_names) == set(kwargs.keys()), 'some unknown meters are passed'
        # Current minibatch stats
        for u, v in kwargs.items():
            self.meter_getter(u, 'mb').add_value(v)
            # Aggregate stats
            self.meter_getter(u, 'ep').update(v, n=mb_size)

        self.lr = lr
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter, mode):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
            mode (str): the mode currently in it.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": mode,
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        for u in self.meter_names:
            stats['{}_{}'.format(u, 'mb')] = float(self.meter_getter(u, 'mb').get_win_median())
            stats['{}_{}'.format(u, 'ep')] = float(self.meter_getter(u, 'ep').avg)

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, mode):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            mode (str): the mode currently in it.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "{}_epoch".format(mode),
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        for u in self.meter_names:
            stats['{}_{}'.format(u, 'ep')] = float(self.meter_getter(u, 'ep').avg)

        logging.log_json_stats(stats)

    def get_avg_for_tb(self):  # This functions prepares and returns for Tensorboard logging
        for u in self.meter_names:
            yield '{}_{}'.format(u, 'ep'), self.meter_getter(u, 'ep').avg

    def get_epoch_loss(self):
        return self.meter_getter('loss', 'ep')

# For a sample meter for multi-view/multi-patch ensemble for testing check out deep_abc
