#!/usr/bin/env python3

from batch_abc import BatchBase
import torch


class Evaluator(BatchBase):
    def __init__(self, cfg, device):
        super().__init__('test' if cfg.TEST.ENABLE else 'valid', cfg, device)

    def set_net_mode(self, net):
        net.eval()

    def _get_lr(self, netwrapper):
        return netwrapper.scheduler.get_last_lr() if self.cfg.SOLVER.SCHEDULER_MODE else self.cfg.SOLVER.BASE_LR

    # @torch.no_grad
    def batch_main(self, netwrapper, x, annotation):
        with torch.no_grad():
            p = netwrapper.forward(x)

            a = self.generate_gt(annotation)

            loss = netwrapper.loss_update(p, a, step=False)

            dice, jaccard, hausdorff = self.evaluate(p, a)

            self.meters.iter_toc()

            self.meters.update_stats(dice, jaccard, hausdorff, loss, self._get_lr(netwrapper), self.cfg.TRAIN.BATCH_SIZE)