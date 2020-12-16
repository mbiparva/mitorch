#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build_test_pipeline import TESTPIPELINE_REGISTRY
import data.transforms_mitorch as tf
from abc import ABC, abstractmethod

__all__ = [
    'NoiseTestTransformations',
    'NoisechannelTestTransformations',
    'ContrastTestTransformations',
    'RotateTestTransformations',
    'ShearTestTransformations',
    'TranslateTestTransformations',
    'ScaleTestTransformations',
    'SpikeTestTransformations',
    'GhostingTestTransformations',
    'BlurTestTransformations',
    'BiasFieldTestTransformations',
    'SwapTestTransformations',
]


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
class NoisechannelTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        sigma = self.params.pop('sigma')
        return tf.AdditiveNoiseChannelWise(sigma=sigma, **self.params)


@TESTPIPELINE_REGISTRY.register()
class ContrastTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        value = self.params.pop('value')
        return tf.RandomContrast(value=value, **self.params)


@TESTPIPELINE_REGISTRY.register()
class RotateTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.AffineRotate(**self.params)


@TESTPIPELINE_REGISTRY.register()
class ShearTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.AffineShear(**self.params)


@TESTPIPELINE_REGISTRY.register()
class TranslateTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.AffineTranslate(**self.params)


@TESTPIPELINE_REGISTRY.register()
class ScaleTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.AffineScale(**self.params)


@TESTPIPELINE_REGISTRY.register()
class SpikeTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.Spike(**self.params)


@TESTPIPELINE_REGISTRY.register()
class GhostingTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.Ghosting(**self.params)


@TESTPIPELINE_REGISTRY.register()
class BlurTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.Blur(**self.params)


@TESTPIPELINE_REGISTRY.register()
class BiasFieldTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.BiasField(**self.params)


@TESTPIPELINE_REGISTRY.register()
class SwapTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.Swap(**self.params)


@TESTPIPELINE_REGISTRY.register()
class MotionTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.PresetMotionArtifact(**self.params)
