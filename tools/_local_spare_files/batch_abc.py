#!/usr/bin/env python3

import time
from abc import ABC, abstractmethod
import datetime
import torch

from data.data_container import DataContainer
from utils.meters import TVTMeter
from utils.metrics import dice_coefficient_metric, jaccard_index_metric, hausdorff_distance_metric


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
        if self.cfg.MODEL.LOSS_FUNC == 'CrossEntropyLoss':
            annotation = annotation.squeeze(dim=1).long()
        return annotation

    @staticmethod
    def cel_prep(p, a):
        p = p.softmax(dim=1)
        p = p[:, 1, ...]
        p = p.unsqueeze(dim=1)
        a = a.float()

        return p, a

    @staticmethod
    def binarize(p, binarize_threshold):
        prediction_mask = p.ge(binarize_threshold)
        p = p.masked_fill(prediction_mask, 1)
        p = p.masked_fill(~prediction_mask, 0)

        return p

    def evaluate(self, p, a):
        BINARIZE_THRESHOLD = 0.5
        if self.cfg.MODEL.LOSS_FUNC == 'CrossEntropyLoss':
            p, a = self.cel_prep(p, a)
        p = self.binarize(p, binarize_threshold=BINARIZE_THRESHOLD)
        return (
            dice_coefficient_metric(p, a, ignore_index=self.cfg.MODEL.IGNORE_INDEX),
            jaccard_index_metric(p, a, ignore_index=self.cfg.MODEL.IGNORE_INDEX),
            hausdorff_distance_metric(p, a, ignore_index=self.cfg.MODEL.IGNORE_INDEX),
        )

    @abstractmethod
    def batch_main(self, net, x, annotation):
        pass

    def batch_loop(self, netwrapper, cur_epoch):

        self.meters.iter_tic()
        for cur_iter, (image, annotation, meta) in enumerate(self.data_container.dataloader):

            image = image.to(self.device, non_blocking=True)
            annotation = annotation.to(self.device, non_blocking=True)

            self.batch_main(netwrapper, image, annotation)

            self.meters.log_iter_stats(cur_epoch, cur_iter, self.mode)

            self.meters.iter_tic()
