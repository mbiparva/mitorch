import os

import torch
import torch.nn as nn
import numpy as np

from models.build import build_model
import utils.checkpoint as checkops
from netwrapper.optimizer import construct_optimizer, construct_scheduler
from netwrapper.build import build_loss


class NetWrapper(nn.Module):
    def __init__(self, device, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net(device)

        self.criterion = self._create_criterion()

        self.optimizer = self._create_optimizer()

        self.scheduler = self._create_scheduler()

    def _create_net(self, device):
        self.net_core = build_model(self.cfg, device)  # this moves to device memory too

    def _create_criterion(self):
        return build_loss(self.cfg)

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

        if step:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
