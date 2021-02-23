#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build_transformations import TRANSFORMATION_REGISTRY
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from abc import ABC, abstractmethod


__all__ = [
    'HFBTransformations',
    'WMHTransformations',
    'NVTTransformations',
    'HPSFTransformations',
]


class BaseTransformations(ABC):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

    def __call__(self, *args, **kwargs):
        return self.create_transform()

    @abstractmethod
    def create_transform(self):
        raise NotImplementedError


@TRANSFORMATION_REGISTRY.register()
class HFBTransformations(BaseTransformations):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def create_transform(self):
        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
                # tf.RandomContrast(value=0.25, prand=True, channel_wise=True),
                # tf.AdditiveNoise(sigma=0.5, noise_type=('gaussian', 'rician', 'rayleigh')[2], randomize_type=False,
                #                  out_of_bound_mode=('normalize', 'clamp')[1], prand=True, channel_wise=True),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
                tf.ResizeImageVolume(self.cfg.DATA.MAX_SIDE_SIZE, min_side=self.cfg.DATA.MIN_SIDE),
                tf.PadToSizeVolume(self.cfg.DATA.MAX_SIDE_SIZE, padding_mode=self.cfg.DATA.PADDING_MODE),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
            tf.NormalizeMeanStdVolume(
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
                inplace=True
            ),
        ]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )


@TRANSFORMATION_REGISTRY.register()
class WMHTransformations(BaseTransformations):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def create_transform(self):
        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            # tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
            # tf.ScaleIntensityRangePercentilesVolume(lower=0.0, upper=99.99, b_min=0, b_max=1,
            #                                         clip=False, relative=False),
            tf.NormalizeMeanStdSingleVolume(nonzero=False, channel_wise=True),
        ]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )


@TRANSFORMATION_REGISTRY.register()
class NVTTransformations(BaseTransformations):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def create_transform(self):
        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),
                # tf.RandomFlipImageVolume(dim=-1),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.RandomCropImageVolumeConditional(self.cfg.DATA.CROP_SIZE, prand=True,
                                                    num_attemps=self.cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                                    threshold=self.cfg.NVT.RANDOM_CROP_THRESHOLD),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            # tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
            # tf.NormalizeMeanStdVolume(
            #     mean=self.cfg.DATA.MEAN,
            #     std=self.cfg.DATA.STD,
            #     inplace=True
            # ),
        ]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )


@TRANSFORMATION_REGISTRY.register()
class HPSFTransformations(BaseTransformations):
    def __init__(self, cfg, mode):
        super().__init__(cfg, mode)

    def create_transform(self):
        # --- BODY ---
        if self.mode == 'train':
            transformations_body = [
                tf.OneHotAnnot(self.cfg.MODEL.NUM_CLASSES),
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
            ]
        elif self.mode in ('valid', 'test'):
            transformations_body = [
                tf.OneHotAnnot(self.cfg.MODEL.NUM_CLASSES),
                tf.ToTensorImageVolume(),
                tf.RandomOrientationTo('RPI'),
                # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
            ]
        else:
            raise NotImplementedError

        # --- TAIL ---
        transformations_tail = [
            tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
        ]

        return torch_tf.Compose(
            transformations_body + transformations_tail
        )
