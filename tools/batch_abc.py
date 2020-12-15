#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import time
from abc import ABC, abstractmethod
import datetime
import torch
import subprocess
import utils.distributed as du
from data.data_container import DataContainer
from utils.meters import TVTMeter
import utils.metrics as metrics
from utils.net_pred import post_proc_pred, pack_pred


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
            self.cfg,
            self.cfg.PROJECT.METERS,
            self.mode,
        )

    def create_dataset(self):
        self.data_container = DataContainer(self.mode, self.cfg)

    def tb_logger_update(self, logger_writer, e):
        if logger_writer is None:  # workers
            return
        for k, m_avg in self.meters.get_avg_for_tb():
            logger_writer.add_scalar('{}/{}'.format(self.mode, k), m_avg, e)

    @abstractmethod
    def set_net_mode(self, net):
        raise NotImplementedError

    def _get_lr(self, netwrapper):
        return netwrapper.scheduler.get_last_lr() if self.cfg.SOLVER.SCHEDULER_MODE else self.cfg.SOLVER.BASE_LR

    def generate_gt(self, annotation):
        if not (self.cfg.HPSF.ENABLE or (self.cfg.NVT.ENABLE and self.cfg.MODEL.NUM_CLASSES > 1)):
            assert annotation.size(1) == 1
        if self.cfg.MODEL.LOSSES[0]['name'] == 'CrossEntropyLoss':
            assert annotation.size(1) == 1
            annotation = annotation.squeeze(dim=1).long()
        if self.cfg.AMP:
            annotation = annotation.to(dtype=torch.float16)
        return annotation

    @staticmethod
    def binarize(p, binarize_threshold):
        prediction_mask = p.ge(binarize_threshold)
        p = p.masked_fill(prediction_mask, 1)
        p = p.masked_fill(~prediction_mask, 0)

        return p

    def ddp_reduce_meters(self, meters):
        # pack
        meters_tensor, meters_keys = list(), list()
        for k, v in meters.items():
            meters_keys.append(k)
            meters_tensor.append(v)
        meters_tensor = torch.tensor(meters_tensor)
        meters_tensor = [meters_tensor.to(self.device)]

        # gather
        meters_tensor_gathered = du.all_reduce(meters_tensor, average=True)

        # unpack
        meters_tensor_gathered = meters_tensor_gathered[0]

        # updated
        for k, v in zip(meters_keys, meters_tensor_gathered.tolist()):
            meters[k] = v

    def evaluate(self, p, a, meters):
        p, a = post_proc_pred(p, a, self.cfg)

        p = self.binarize(p, binarize_threshold=self.cfg.TRAIN.BINARIZE_THRESHOLD)

        for m in self.cfg.PROJECT.METERS:
            if m == 'loss':
                continue
            metric_function = getattr(metrics, f'{m}_metric')
            meters[m] = metric_function(p, a, ignore_index=self.cfg.MODEL.IGNORE_INDEX)

        # do all_reduce (sum) to sync meters across processes
        if self.cfg.DDP:
            self.ddp_reduce_meters(meters)

    @staticmethod
    def depth_sampling(image, annotation):
        # create index tensor
        image_shape = torch.tensor(image.shape)
        d_size = image_shape[2].item()  # depth size
        image_shape[2] = 1  # depth size is set to 1
        b_size, image_shape = image_shape[0], image_shape[1:]
        image_numel = image_shape.prod().item()

        rand_ind = torch.randint(d_size, (b_size, 1))
        rand_ind = rand_ind.repeat(1, image_numel)
        rand_ind = rand_ind.reshape((b_size, *image_shape))
        rand_ind = rand_ind.to(image.device)

        # gather
        image = image.gather(dim=2, index=rand_ind)

        rand_ind = rand_ind[:, -1:, :]
        annotation = annotation.gather(dim=2, index=rand_ind)

        return image, annotation

    def batch_main(self, netwrapper, x, annotation, step=True):
        meters = dict()

        if self.cfg.WMH.ENABLE:
            p, annotation = netwrapper.forward((x, annotation))
        else:
            p = netwrapper.forward(x)

        a = self.generate_gt(annotation)

        p = pack_pred(p)

        meters['loss'] = netwrapper.loss_update(p, a, step=step)

        self.evaluate(p, a, meters)

        self.meters.iter_toc()

        self.meters.update_stats(self._get_lr(netwrapper), self.cfg.TRAIN.BATCH_SIZE, **meters)

    def batch_main_mode(self, netwrapper, x, annotation):
        if self.mode == 'train':
            return self.batch_main(netwrapper, x, annotation, step=True)
        elif self.mode == 'valid':
            with torch.no_grad():
                return self.batch_main(netwrapper, x, annotation, step=False)
        else:
            raise NotImplementedError

    def batch_loop(self, netwrapper, cur_epoch):

        self.meters.iter_tic()
        for cur_iter, (image, annotation, meta) in enumerate(self.data_container.dataloader):
            if cur_epoch == 0 and cur_iter % 50 == 0:
                if self.cfg.DDP:
                    if not self.cfg.DDP_CFG.RANK:
                        subprocess.call(['nvidia-smi'])
                else:
                    subprocess.call(['nvidia-smi'])

            image = image.to(self.device, non_blocking=True)
            annotation = annotation.to(self.device, non_blocking=True)

            # For now I will keep 2d depth sampling at system-level, for both train and val, later could exclude val
            if self.cfg.MODEL.PROCESSING_MODE == '2d':
                image, annotation = self.depth_sampling(image, annotation)

            self.batch_main_mode(netwrapper, image, annotation)

            self.meters.log_iter_stats(cur_epoch, cur_iter, self.mode)

            self.meters.iter_tic()
