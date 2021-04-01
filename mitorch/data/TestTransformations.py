#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build_test_pipeline import TESTPIPELINE_REGISTRY
import data.transforms_mitorch as tf
from abc import ABC, abstractmethod
import torch

__all__ = [
    'NoiseTestTransformations',
    'NoisechannelTestTransformations',
    'ContrastTestTransformations',
    'ContrastchannelTestTransformations',
    'GammaTestTransformations',
    'RotateTestTransformations',
    'ShearTestTransformations',
    'TranslateTestTransformations',
    'ScaleTestTransformations',
    'SpikeTestTransformations',
    'GhostingTestTransformations',
    'MotionTestTransformations',
    'BlurTestTransformations',
    'BiasfieldTestTransformations',
    'SwapTestTransformations',
    'AnisotropyTestTransformations',
    'ElasticdeformationTestTransformations',
    'ZoomTestTransformations',
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
class ContrastchannelTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        value = self.params.pop('value')
        return tf.RandomContrastChannelWise(value=value, **self.params)


@TESTPIPELINE_REGISTRY.register()
class GammaTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        value = self.params.pop('value')
        self.params['channel_wise'] = True
        self.params['prand'] = False
        return tf.RandomGamma(value=value, **self.params)


# noinspection PyTypeChecker
@TESTPIPELINE_REGISTRY.register()
class RotateTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        self.params['padding_mode'] = 'zeros'
        radian = self.params.pop('radian')
        radian_ls = [0]*3
        radian_ls[torch.randint(0, 3, size=[1]).item()] = radian
        self.params['rotate_params'] = radian_ls
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
        return tf.RandomSpike(**self.params)


@TESTPIPELINE_REGISTRY.register()
class GhostingTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        self.params['restore'] = 0.5
        self.params['axis'] = torch.randint(0, 3, size=[1]).item()
        self.params['num_ghosts'] = int(self.params['num_ghosts'])
        return tf.Ghosting(**self.params)


@TESTPIPELINE_REGISTRY.register()
class BlurTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        return tf.Blur(**self.params)


@TESTPIPELINE_REGISTRY.register()
class BiasfieldTestTransformations(BaseTransformations):
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
        self.params['image_interpolation'] = 'bspline'
        return tf.MotionVolume(**self.params)


@TESTPIPELINE_REGISTRY.register()
class AnisotropyTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        downsampling = self.params.pop('downsampling')
        axes = (0, 1, 2)  # torch.randint(0, 3, size=[1]).item()
        self.params['interpolation_mode'] = 'trilinear'
        return tf.AnisotropyVolume(axes, downsampling, **self.params)


@TESTPIPELINE_REGISTRY.register()
class ElasticdeformationTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        max_displacement = [self.params.pop('md')]*3
        num_control_points = 7
        self.params['image_interpolation'] = 'bspline'
        return tf.ElasticDeformationVolume(num_control_points, max_displacement, **self.params)


@TESTPIPELINE_REGISTRY.register()
class ZoomTestTransformations(BaseTransformations):
    def __init__(self, cfg, params):
        super().__init__(cfg, params)

    def create_transform(self):
        factor = self.params.pop('factor')
        self.params['mode'] = 'trilinear'
        self.params['padding_mode'] = 'constant'
        return tf.ZoomVolume(zoom=factor, **self.params)
