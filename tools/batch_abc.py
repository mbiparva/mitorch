#!/usr/bin/env python3

import time
from abc import ABC, abstractmethod
import datetime
import torch

from data.data_container import DataContainer
from utils.meters import TrainMeter, ValMeter, TestMeter
from fastai.metrics import accuracy, top_k_accuracy


class BatchBase(ABC):
    modes = ('train', 'valid', 'test')

    def __init__(self, mode, cfg, device):
        self.cfg = cfg
        self.data_container = None
        assert mode in BatchBase.modes
        self.mode, self.device = mode, device

        self.create_dataset()

        self.meters = self.create_meters()

    def create_meters(self):
        meters = (TrainMeter, ValMeter, ValMeter)[BatchBase.modes.index(self.mode)]  # TestMeter for now is ValidMeter
        return meters(
            len(self.data_container.dataloader),
            self.cfg
        )

    def create_dataset(self):
        self.data_container = DataContainer(self.mode, self.cfg)

    def tb_logger_update(self, logger_writer, e):
        for k, m_avg in self.meters.get_avg_for_tb():
            logger_writer.add_scalar('{}/{}'.format(self.mode, k), m_avg, e)

    @abstractmethod
    def set_net_mode(self, net):
        pass

    @staticmethod
    def generate_gt(annotation):
        return annotation

    @staticmethod
    def evaluate(p, a):
        return accuracy(p, a).item(), top_k_accuracy(p, a, 5).item()

    @abstractmethod
    def batch_main(self, net, x, annotation):
        pass

    def batch_loop(self, netwrapper, cur_epoch):

        self.meters.iter_tic()
        for cur_iter, (image, annotation) in enumerate(self.data_container.dataloader):

            image = image.to(self.device, non_blocking=True)
            annotation = annotation.to(self.device, non_blocking=True)

            self.batch_main(netwrapper, image, annotation)

            self.meters.log_iter_stats(cur_epoch, cur_iter)

            self.meters.iter_tic()
