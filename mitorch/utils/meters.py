#!/usr/bin/env python3

"""Meters."""

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer

import utils.logging as logging
import utils.metrics as metrics
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

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.dice_mb = ScalarMeter(cfg.LOG_PERIOD)
        self.jaccard_mb = ScalarMeter(cfg.LOG_PERIOD)
        self.hausdorff_mb = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_mb = ScalarMeter(cfg.LOG_PERIOD)
        # Epoch stats
        self.dice_total = AverageMeter()
        self.jaccard_total = AverageMeter()
        self.hausdorff_total = AverageMeter()
        self.loss_total = AverageMeter()
        self.lr = None
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.dice_mb.reset()
        self.jaccard_mb.reset()
        self.hausdorff_mb.reset()
        self.loss_mb.reset()
        self.dice_total.reset()
        self.jaccard_total.reset()
        self.hausdorff_total.reset()
        self.loss_total.reset()
        self.lr = None
        self.num_samples = 0

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

    def update_stats(self, dice, jaccard, hausdorff, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            dice (float): dice coefficient.
            jaccard (float): jaccard index.
            hausdorff (float): hausdorff distance.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.dice_mb.add_value(dice)
        self.jaccard_mb.add_value(jaccard)
        self.hausdorff_mb.add_value(hausdorff)
        self.loss_mb.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.dice_total.update(dice, n=mb_size)
        self.jaccard_total.update(jaccard, n=mb_size)
        self.hausdorff_total.update(hausdorff, n=mb_size)
        self.loss_total.update(loss, n=mb_size)
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
            "dice_mb": self.dice_mb.get_win_median(),
            "jaccard_mb": self.jaccard_mb.get_win_median(),
            "hausdorff_mb": self.hausdorff_mb.get_win_median(),
            "loss_mb": self.loss_mb.get_win_median(),
            "dice_ep": self.dice_total.avg,
            "jaccard_ep": self.jaccard_total.avg,
            "hausdorff_ep": self.hausdorff_total.avg,
            "loss_ep": self.loss_total.avg,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        dice_coeff = self.dice_total.avg
        jaccard_index = self.jaccard_total.avg
        hausdorff_distance = self.hausdorff_total.avg
        loss = self.loss_total.avg
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "dice_coeff": dice_coeff,
            "jaccard_index": jaccard_index,
            "hausdorff_distance": hausdorff_distance,
            "loss": loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def get_avg_for_tb(self):  # This functions prepares and returns for Tensorboard logging
        meters = {
            'loss': self.loss_total.avg,
            'dice_coeff': self.dice_total.avg,
            'jaccard_index': self.jaccard_total.avg,
            'hausdorff_distance': self.hausdorff_total.avg,
        }
        for k, m in meters.items():
            yield k, m

    def get_epoch_loss(self):
        return self.loss_total.avg

# For a sample meter for multi-view/multi-patch ensemble for testing check out deep_abc
