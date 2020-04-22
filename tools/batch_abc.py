#!/usr/bin/env python3

import time
from abc import ABC, abstractmethod
import datetime
import torch

from data.data_container import DataContainer
from utils.meters import TVTMeter
from utils.metrics import dice_coefficient_metric


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
        meters = TVTMeter
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
        raise NotImplementedError

    def generate_gt(self, annotation):
        assert annotation.size(1) == 1
        if self.cfg.MODEL.LOSS_FUNC == 'DiceLoss':
            annotation = annotation.unsqueeze(dim=1)
            annotation = annotation.to(dtype=torch.float)
        return annotation.squeeze(dim=1).long()

    def evaluate(self, p, a):
        # method = ('own', 'fastai')[0]
        if self.cfg.MODEL.LOSS_FUNC == 'CrossEntropyLoss':
            p = p.softmax(dim=1)
            p = p[:, 1, ...]
            # true for p.ndim==5 and a.ndim==4 --- check other scenarios
            assert p.ndim == a.ndim == 4
            p = p.unsqueeze(dim=1)
        # a = a.unsqueeze(dim=1)
        # if method == 'fastai':
        #     from fastai.metrics import dice
        #     return dice(p, a, iou=False, eps=1e-6).item()
        # else:
        return dice_coefficient_metric(p, a, ignore_index=self.cfg.MODEL.IGNORE_INDEX, threshold=0.5)

    @abstractmethod
    def batch_main(self, net, x, annotation):
        pass

    def batch_loop(self, netwrapper, cur_epoch):

        self.meters.iter_tic()
        for cur_iter, (image, annotation, meta) in enumerate(self.data_container.dataloader):

            image = image.to(self.device, non_blocking=True)
            annotation = annotation.to(self.device, non_blocking=True)

            self.batch_main(netwrapper, image, annotation)

            self.meters.log_iter_stats(cur_epoch, cur_iter)

            self.meters.iter_tic()
