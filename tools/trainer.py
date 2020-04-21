#!/usr/bin/env python3

from batch_abc import BatchBase


class Trainer(BatchBase):
    def __init__(self, cfg, device):
        super().__init__('train', cfg, device)

    def set_net_mode(self, net):
        net.train()

    def _get_lr(self, netwrapper):
        return netwrapper.scheduler.get_last_lr() if self.cfg.SOLVER.SCHEDULER_MODE else self.cfg.SOLVER.BASE_LR

    def batch_main(self, netwrapper, x, annotation):
        p = netwrapper.forward(x)

        a = self.generate_gt(annotation)

        loss = netwrapper.loss_update(p, a, step=True)

        acc, acc5 = self.evaluate(p, a)

        self.meters.iter_toc()

        self.meters.update_stats(acc, acc5, loss, self._get_lr(netwrapper), self.cfg.TRAIN.BATCH_SIZE)
