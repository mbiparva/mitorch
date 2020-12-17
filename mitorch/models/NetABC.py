#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from abc import ABC, abstractmethod
import torch.nn as nn
from models.weight_init_helper import init_weights
try:
    # noinspection PyUnresolvedReferences
    from torch.cuda.amp import autocast
except ImportError:
    pass


class NetABC(ABC, nn.Module):
    def __init__(self, cfg, auto_init=True):
        super().__init__()

        self.cfg = cfg.clone()

        self.set_processing_mode()

        self._create_net()

        if auto_init:
            self.init_weights()

    @abstractmethod
    def set_processing_mode(self):
        raise NotImplementedError

    @abstractmethod
    def _create_net(self):
        raise NotImplementedError

    def init_weights(self):
        init_weights(self, self.cfg.MODEL.FC_INIT_STD)

    @abstractmethod
    def forward_core(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self.cfg.AMP:
            with autocast():
                return self.forward_core(x)
        return self.forward_core(x)
