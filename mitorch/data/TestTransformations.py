#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build_test_pipeline import TESTPIPELINE_REGISTRY
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from abc import ABC, abstractmethod


class BaseTransformations(ABC):
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.params = params

        self.transform = self.create_transform()

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def create_transform(self):
        raise NotImplementedError


@TESTPIPELINE_REGISTRY.register()
class NoiseTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        sigma = self.params.pop('sigma')
        return tf.AdditiveNoise(sigma=sigma, **self.params)


@TESTPIPELINE_REGISTRY.register()
class ContrastTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        value = self.params.pop('value')
        return tf.RandomContrast(value=value, **self.params)
