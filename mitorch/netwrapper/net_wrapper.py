#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os

import torch
import torch.nn as nn
import numpy as np

from models.build import build_model
import utils.checkpoint as checkops
from netwrapper.optimizer import construct_optimizer, construct_scheduler
from netwrapper.build import build_loss
from data.functional_mitorch import resize, pad


class NetWrapper(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net(device)

        self.criterion = self._create_criterion()
        self.criterion_aux = self._create_criterion('WeightedHausdorffLoss') if self.cfg.MODEL.LOSS_AUG_WHL else None

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()

    def _create_net(self, device):
        self.net_core = build_model(self.cfg, device)  # this moves to device memory too

    def _create_criterion(self, name=None):
        return build_loss(self.cfg, name)

    def _create_optimizer(self):
        return construct_optimizer(self.net_core, self.cfg)

    def _create_scheduler(self):
        return construct_scheduler(self.optimizer, self.cfg)

    def schedule_step(self, eval_meter_avg=None):
        if self.cfg.SOLVER.SCHEDULER_MODE:
            if self.cfg.SOLVER.SCHEDULER_TYPE in ('plateau', ):
                assert eval_meter_avg is not None, 'for learning scheduler PLATEAU, you must turn on validation'
                self.scheduler.step(eval_meter_avg)
            else:
                self.scheduler.step()

    def save_checkpoint(self, ckpnt_path, cur_epoch):
        checkops.save_checkpoint(
            ckpnt_path,
            self.net_core,
            self.optimizer,
            cur_epoch,
            self.cfg
        )

    def load_checkpoint(self, ckpnt_path):
        return checkops.load_checkpoint(
            ckpnt_path,
            self.net_wrapper.net_core,
            self.cfg.NUM_GPUS > 1,
            self.net_wrapper.optimizer
        )

    def forward(self, x):
        x = self.net_core(x)

        return x

    def loss_update(self, p, a, step=True):
        loss = self.criterion(p, a)
        if self.cfg.MODEL.LOSS_AUG_WHL:
            loss += 0.1 * self.criterion_aux(p, a)

        if step:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()


class NetWrapperHFB(NetWrapper):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)


class NetWrapperWMH(NetWrapper):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)

    def _create_net(self, device):
        self._create_net_hfb(device)
        self.net_core = build_model(self.cfg, device)

    def _create_net_hfb(self, device):
        # REMOVE IT
        N_BASE_FILTERS = self.cfg.MODEL.N_BASE_FILTERS
        ENCO_DEPTH = self.cfg.MODEL.ENCO_DEPTH
        DROPOUT_RATE = self.cfg.MODEL.DROPOUT_RATE
        self.cfg.MODEL.N_BASE_FILTERS = 16
        self.cfg.MODEL.ENCO_DEPTH = 5
        self.cfg.MODEL.DROPOUT_RATE = 0.25
        # REMOVE IT

        self.net_core_hfb = build_model(self.cfg, device)
        self.load_checkpoint_hfb(self.cfg.WMH.HFB_CHECKPOINT)
        self.net_core_hfb.eval()

        # REMOVE IT
        self.cfg.MODEL.N_BASE_FILTERS = N_BASE_FILTERS
        self.cfg.MODEL.ENCO_DEPTH = ENCO_DEPTH
        self.cfg.MODEL.DROPOUT_RATE = DROPOUT_RATE
        # REMOVE IT

    def load_checkpoint_hfb(self, ckpnt_path):
        checkops.load_checkpoint(ckpnt_path, self.net_core_hfb, data_parallel=self.cfg.NUM_GPUS > 1)

    def load_checkpoint(self, ckpnt_path):
        self.load_checkpoint_hfb(self.cfg.WMH.HFB_CHECKPOINT)
        return checkops.load_checkpoint(
            ckpnt_path,
            self.net_wrapper.net_core,
            self.cfg.NUM_GPUS > 1,
            self.net_wrapper.optimizer
        )

    def forward(self, x):
        x, annotation = self.hfb_extract(x)
        x = self.net_core(x)

        return x, annotation

    @staticmethod
    def binarize_pred(p, binarize_threshold):
        prediction_mask = p.ge(binarize_threshold)
        p = p.masked_fill(prediction_mask, 1)
        p = p.masked_fill(~prediction_mask, 0)

        return p

    @staticmethod
    def gen_cropping_box(pred):
        return [(p.eq(1).nonzero().min(0)[0].tolist(), p.eq(1).nonzero().max(0)[0].tolist()) for p in pred]

    def crop_masked_input(self, x, cropping_box):
        b, _, d, h, w = x.shape
        if self.cfg.WMH.CROPPING:
            pad_amount = 2

            def pad_lower_clamp(value):
                return max(0, value - pad_amount)

            def pad_upper_clamp(value):
                return min(d, value + pad_amount)

            return [
                x[
                    i,
                    :,
                    pad_lower_clamp(cropping_box[i][0][0]): pad_upper_clamp(cropping_box[i][1][0]),  # max is inclusive
                    pad_lower_clamp(cropping_box[i][0][1]): pad_upper_clamp(cropping_box[i][1][1]),
                    pad_lower_clamp(cropping_box[i][0][2]): pad_upper_clamp(cropping_box[i][1][2])
                ] for i in range(b)
            ]
        else:
            return [
                x[i, :] for i in range(b)
            ]

    @staticmethod
    def pad_input(x, target_size, fill, padding_mode):
        target_size = torch.tensor(tuple([target_size] * 3))
        target_size = target_size.clone()
        auto_fill_ind = target_size == -1
        image_size = torch.tensor(x.shape[1:])
        target_size[auto_fill_ind] = image_size[auto_fill_ind]
        assert (image_size <= target_size).all()
        size_offset = target_size - image_size
        padding_before = size_offset // 2
        padding_after = size_offset - padding_before
        padding = tuple(torch.stack((padding_before.flip(0), padding_after.flip(0))).T.flatten().tolist())

        if padding_mode in ('mean', 'median', 'min', 'max'):
            fill = getattr(x, padding_mode)().item()
            padding_mode = 'constant'

        return pad(x, padding, fill, padding_mode)

    def resize_pad_input(self, x, target_size, fill, padding_mode, interpolation='trilinear'):
        if self.cfg.WMH.RESIZING_PADDING:
            # x = [resize(v, [target_size]*3, interpolation, min_side=False) for v in x]
            x = [resize(v, target_size, interpolation, min_side=False) for v in x]
            x = [self.pad_input(v, target_size, fill, padding_mode) for v in x]

        x = torch.stack(x)

        return x

    def compute_pred(self, x):
        # predict mask
        with torch.no_grad():
            pred = self.net_core_hfb(x)

        # generate mask
        pred = self.binarize_pred(pred, binarize_threshold=self.cfg.WMH.BINARIZE_THRESHOLD)

        pred = pred.squeeze(1)

        return pred

    def resize_crop_pad_input(self, x, cropping_box):
        # crop masked input using the cropping_box
        x = self.crop_masked_input(x, cropping_box)

        # resize cropped input
        x = self.resize_pad_input(x, self.cfg.WMH.MAX_SIDE_SIZE, self.cfg.WMH.FILL, self.cfg.WMH.PADDING_MODE)

        return x

    def resize_crop_pad_annot(self, annotation, cropping_box):
        annotation = self.crop_masked_input(annotation, cropping_box)
        annotation = self.resize_pad_input(
            annotation,
            self.cfg.WMH.MAX_SIDE_SIZE,
            0,
            'constant',
            interpolation='nearest'
        )

        return annotation

    def hfb_extract(self, x):
        x, annotation = x

        if self.cfg.WMH.HFB_GT:
            annotation, pred = annotation[:, 0], annotation[:, 1]
        else:
            annotation = annotation.squeeze(1)
            pred = self.compute_pred(x)
        annotation = annotation.unsqueeze(1)

        # generate cropping_box
        cropping_box = self.gen_cropping_box(pred)

        # multiply input with binary mask
        x = x * pred  # TODO we can ignore masking out-of-mask regions out

        # crop and resize-pad input
        x = self.resize_crop_pad_input(x, cropping_box)

        # crop and resize-pad annotation
        annotation = self.resize_crop_pad_annot(annotation, cropping_box)

        return x, annotation
