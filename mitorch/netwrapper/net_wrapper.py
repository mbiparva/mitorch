#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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
from data.build_transformations import build_transformations
from models.Unet3D import pad_if_necessary
try:
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_SUPPORTED = True
except ImportError:
    AMP_SUPPORTED = False
    print('AMP is not supported in this environment')


class NetWrapper(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg
        self.grad_scaler = None

        self._create_net(device)

        self.criteria = self._create_criterion_multi()

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()

    def _create_net(self, device):
        self.net_core = build_model(self.cfg, device)  # this moves to device memory too

        if self.cfg.AMP:
            self.grad_scaler = GradScaler()

    def _create_criterion(self, name, with_logits):
        return build_loss(self.cfg, name, with_logits)

    def _create_criterion_multi(self):
        losses = [
            (loss['weight'], self._create_criterion(loss['name'], loss['with_logits']))
            for loss in self.cfg.MODEL.LOSSES
        ]

        return losses

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

    def save_checkpoint(self, ckpnt_path, cur_epoch, best=False):
        checkops.save_checkpoint(
            ckpnt_path,
            self.net_core,
            self.optimizer,
            cur_epoch,
            self.cfg,
            scaler=self.grad_scaler,
            best=best
        )

    def load_checkpoint(self, ckpnt_path):
        return checkops.load_checkpoint(
            ckpnt_path,
            self.net_wrapper.net_core,
            self.cfg.DDP,
            self.net_wrapper.optimizer,
            scaler=self.grad_scaler,
        )

    def forward(self, x):
        x = self.net_core.forward(x)

        return x

    def compute_loss_core(self, p, a):
        loss = torch.tensor([0], dtype=torch.float, device=p[0].device)
        a = [a] * len(p)
        for p_i, a_i in zip(p, a):
            for weight, criterion in self.criteria:
                loss += weight * criterion(p_i, a_i)

        return loss

    def compute_loss(self, p, a):
        if self.cfg.AMP:
            with autocast():
                return self.compute_loss_core(p, a)
        return self.compute_loss_core(p, a)

    def loss_update(self, p, a, step=True):
        loss = self.compute_loss(p, a)

        if step:
            if self.cfg.AMP:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()

                return loss.item()

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
        if not self.cfg.WMH.HFB_GT:
            self._create_net_hfb(device)
        self.net_core = build_model(self.cfg, device)

        if self.cfg.AMP:
            self.grad_scaler = GradScaler()

    def _create_net_hfb(self, device):
        self.net_core_hfb = build_model(self.cfg, device)
        self.load_checkpoint_hfb(self.cfg.WMH.HFB_CHECKPOINT)
        self.net_core_hfb.eval()

    def load_checkpoint_hfb(self, ckpnt_path):
        checkops.load_checkpoint(ckpnt_path, self.net_core_hfb,
                                 distributed_data_parallel=self.cfg.DDP)

    def load_checkpoint(self, ckpnt_path):
        self.load_checkpoint_hfb(self.cfg.WMH.HFB_CHECKPOINT)
        return checkops.load_checkpoint(
            ckpnt_path,
            self.net_wrapper.net_core,
            self.cfg.DDP,
            self.net_wrapper.optimizer
        )

    def forward(self, x, return_input=False):
        assert self.cfg.MODEL.PROCESSING_MODE == '3d', '2d processing is not addressed for WMH and HFB'

        x, annotation = self.hfb_extract(x)

        pred = self.net_core(x)

        if isinstance(pred, list):
            for i in range(len(pred)):
                pred[i], annotation = pad_if_necessary(pred[i], annotation)
        else:
            pred, annotation = pad_if_necessary(pred, annotation)

        if return_input:
            return pred, annotation, x
        else:
            return pred, annotation

    @staticmethod
    def binarize_pred(p, binarize_threshold):
        prediction_mask = p.ge(binarize_threshold)
        p = p.masked_fill(prediction_mask, 1)
        p = p.masked_fill(~prediction_mask, 0)

        return p

    @staticmethod
    def gen_cropping_box(pred):
        return [
            (p.eq(1).nonzero(as_tuple=False).min(0)[0].tolist(), p.eq(1).nonzero(as_tuple=False).max(0)[0].tolist())
            for p in pred
        ]

    def crop_masked_input(self, x, cropping_box):
        b, _, d, h, w = x.shape
        if self.cfg.WMH.CROPPING:
            pad_amount = 1

            def pad_lower_clamp(value, m_value=0):
                return max(m_value, value - pad_amount)

            def pad_upper_clamp(value, m_value):
                return min(m_value, value + pad_amount)

            return [
                x[
                    i,
                    :,
                    pad_lower_clamp(cropping_box[i][0][0]): pad_upper_clamp(cropping_box[i][1][0], d) + 1,
                    pad_lower_clamp(cropping_box[i][0][1]): pad_upper_clamp(cropping_box[i][1][1], h) + 1,
                    pad_lower_clamp(cropping_box[i][0][2]): pad_upper_clamp(cropping_box[i][1][2], w) + 1
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

    def unpack_predict_mask(self, x):
        x, annotation = x

        if self.cfg.WMH.HFB_GT:
            annotation, pred = annotation[:, 0], annotation[:, 1]
        else:
            annotation = annotation.squeeze(1)
            pred = self.compute_pred(x)
        annotation = annotation.unsqueeze(1)
        # save_pred(pred.unsqueeze(1), save_dir, time_id+'_hfb', *[x])
        # save_pred(annotation, save_dir, time_id+'_wmh')

        return x, pred, annotation

    def hfb_extract_manual(self, x, pred, annotation):
        # REMOVE IT
        # from test_net import save_pred
        # import time
        # time_id = str(int(time.time()))
        # save_dir = '/gpfs/fs0/scratch/m/mgoubran/mbiparva/wmh_pytorch/tools/samples'

        # generate cropping_box
        cropping_box = self.gen_cropping_box(pred)

        # multiply input with binary mask
        x = x * pred  # TODO we can ignore masking out-of-mask regions out

        # crop and resize-pad input
        x = self.resize_crop_pad_input(x, cropping_box)

        # crop and resize-pad annotation
        annotation = self.resize_crop_pad_annot(annotation, cropping_box)

        # REMOVE IT
        # save_pred(annotation, save_dir, time_id, *[x])

        return x, annotation

    def hfb_extract_pipeline(self, x, pred, annotation):
        if not self.cfg.WMH.HFB_GT:
            raise NotImplementedError('check me if you want to use this, it might need to get updated')

            hfb_transformations = build_transformations('WMHSkullStrippingTransformations', self.cfg, 'train')()

            x_annotation = torch.stack((x, annotation), dim=1)

            x_annotation, _, _ = hfb_transformations((x_annotation, pred, None))  # meta is None

            x, annotation = x_annotation[:, :-1], x_annotation[:, -1]

        return x, annotation

    def hfb_extract(self, x):
        x, pred, annotation = self.unpack_predict_mask(x)

        if self.cfg.WMH.HFB_MASKING_MODE == 'manual':
            return self.hfb_extract_manual(x, pred, annotation)
        else:
            return self.hfb_extract_pipeline(x, pred, annotation)
