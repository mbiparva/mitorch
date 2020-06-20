#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from batch_abc import BatchBase


class Trainer(BatchBase):
    def __init__(self, cfg, device):
        super().__init__('train', cfg, device)

    def set_net_mode(self, net):
        net.train()

    def batch_main(self, netwrapper, x, annotation):
        meters = dict()

        if self.cfg.WMH.ENABLE:
            p, annotation = netwrapper.forward((x, annotation))
        else:
            p = netwrapper.forward(x)

        a = self.generate_gt(annotation)

        meters['loss'] = netwrapper.loss_update(p, a, step=True)

        self.evaluate(p, a, meters)

        self.meters.iter_toc()

        self.meters.update_stats(self._get_lr(netwrapper), self.cfg.TRAIN.BATCH_SIZE, **meters)
