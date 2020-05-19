#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)


import torch
import numpy as np
import warnings
from typing import Hashable
from abc import ABC, abstractmethod


class Transformable(ABC):

    @abstractmethod
    def __call__(self, volume, *args, **kwargs):
        raise NotImplementedError


class Randomizable(Transformable):
    def __init__(self):
        self.srand_thr = 0
        self.srand_p = 1

    @staticmethod
    def set_random_state(seed=None, state=None):
        if seed is not None:
            torch.random.manual_seed(seed)

        if state is not None:
            torch.random.set_rng_state(state)

    def randomize(self, *args, **kwargs):
        if self.srand_thr > 0:
            self.srand_p = torch.rand(1).item()
        self.randomize_params()

    @abstractmethod
    def randomize_params(self):
        raise NotImplementedError

    def __call__(self, volume, *args, **kwargs):
        self.randomize()

        if self.srand_p < self.srand_thr:
            return volume

        self.apply(volume, *args, **kwargs)

    @abstractmethod
    def apply(self, volume, *args, **kwargs):
        raise NotImplementedError
