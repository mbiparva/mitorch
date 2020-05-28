#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from batch_abc import BatchBase
import torch


class Evaluator(BatchBase):
    def __init__(self, cfg, device):
        super().__init__('test' if cfg.TEST.ENABLE else 'valid', cfg, device)

    def set_net_mode(self, net):
        net.eval()

    @torch.no_grad()
    def batch_main(self, netwrapper, x, annotation):
        meters = dict()

        p = netwrapper.forward(x)

        a = self.generate_gt(annotation)

        meters['loss'] = netwrapper.loss_update(p, a, step=False)

        self.evaluate(p, a, meters)

        self.meters.iter_toc()

        self.meters.update_stats(self._get_lr(netwrapper), self.cfg.TRAIN.BATCH_SIZE, **meters)
