#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)


import torch
import numpy as np
import warnings
from typing import Hashable
from abc import ABC, abstractmethod


class Transformable(ABC):
    def __call__(self, volume):
        return self.apply(volume)

    @abstractmethod
    def apply(self, volume):
        raise NotImplementedError


class Randomizable(Transformable):
    def __init__(self, prand=False, **kwargs):
        """
        Sets the general randomization status of the transformation
        Args:
            prand (bool): parameters randomization mode (default False). If it is False,
                parameters are never randomized so it is like a regular transformation. Otherwise, parameters are
                automatically randomized every time it is called.
        """
        assert isinstance(prand, bool)
        self.prand = prand

    @staticmethod
    def set_random_state(seed=None, state=None):
        if seed is not None:
            torch.random.manual_seed(seed)
        if state is not None:
            torch.random.set_rng_state(state)

    def randomize(self, volume):
        if self.prand:
            self.randomize_params(volume)

    @abstractmethod
    def randomize_params(self, volume):
        raise NotImplementedError

    def __call__(self, volume):
        self.randomize(volume)

        return self.apply(volume)

    @abstractmethod
    def apply(self, volume):
        raise NotImplementedError
