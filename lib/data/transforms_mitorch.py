"""
Implemented by Mahdi Biparva, April 2020 @ Sunnybrook Research Institure (SRI) - Brain Imaging Lab (BIL)
Inspired from the torchvision transform files for video processing.
This contains a list of transformation modules to process image volumes.
Image volumes represent dense 3D volumes generated from CT/MRI scans.
"""

import torch
import numbers
import random
from PIL import Image
from torchvision.transforms import (
    RandomCrop,
    RandomResizedCrop,
)
from . import functional_mitorch as F
import collections
import sys

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


__all__ = [
    "RandomCropImageVolume",
    "RandomResizedCropImageVolume",
    "CenterCropImageVolume",
    "NormalizeImageVolume",
    "ToTensorImageVolume",
    "RandomFlipImageVolume",
    'ResizeImageVolume',
    'OrientationToRAI',
    'ResampleTo1mm',

]


class OrientationToRAI(object):
    def __init__(self):
        pass

    def __call__(self, volume):
        image, annot, meta = volume
        direction = torch.tensor(meta['direction'], dtype=torch.float)
        direction_diagonal = direction.reshape(3, 3).diagonal()
        direction_sign = direction_diagonal.sign()
        assert (direction_sign.abs() == 1).all().item()
        for i, d in enumerate(direction_sign):
            if d > 0:
                continue
            image = F.flip(image, i)
            annot = F.flip(annot, i)
            direction_diagonal *= -1
        meta['direction'] = tuple(direction.tolist())
        return (
            image,
            annot,
            meta
        )


class ResampleTo1mm(object):
    def __init__(self, interpolation='trilinear'):
        self.interpolation = interpolation

    def __call__(self, volume):
        image, annot, meta = volume
        size = torch.tensor(meta['size'], dtype=torch.float)
        spacing = torch.tensor(meta['spacing'], dtype=torch.float)
        iso1mm = torch.tensor([1]*3, dtype=torch.float)
        if (spacing == iso1mm).all().item():
            return volume
        size = (size * spacing).floor().int().tolist()[::-1]  # reverse size since F.resize works in DxHxW space
        image, annot = (
            F.resize(image, size, self.interpolation),
            F.resize(annot, size, 'nearest'),
        )
        meta['size'] = size[::-1]
        meta['spacing'] = (1, 1, 1)

        return image, annot, meta


# noinspection PyMissingConstructor,PyTypeChecker
class RandomCropImageVolume(RandomCrop):
    def __init__(self, size):
        raise NotImplementedError
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image, mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, OH, OW)
        """
        image, annot, meta = volume
        i, j, h, w = self.get_params(image, self.size)
        return (
            F.crop(image, i, j, h, w),
            F.crop(annot, i, j, h, w),
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


# noinspection PyMissingConstructor
class RandomResizedCropImageVolume(RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='trilinear'):
        raise NotImplementedError
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, H, W)
        """
        image, annot, meta = volume
        i, j, h, w = self.get_params(volume, self.scale, self.ratio)
        return (
            F.resized_crop(image, i, j, h, w, self.size, self.interpolation),
            F.resized_crop(annot, i, j, h, w, self.size, 'nearest'),
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2}, ratio={3})'.format(
                self.size, self.interpolation, self.scale, self.ratio
            )


# noinspection PyTypeChecker
class ResizeImageVolume(object):
    """Resize the input image volume to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (d, h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if depth > height > width, then image will be rescaled to
            (size * depth / width, size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is trilinear
    """

    def __init__(self, size=None, scale_factor=None, interpolation='trilinear'):
        assert size or scale_factor, 'either size or scale_factor must be given'
        if size:
            assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 3)
        if scale_factor:
            assert isinstance(scale_factor, float)
        self.scale_factor = scale_factor
        self.size = size
        self.interpolation = interpolation

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be resized. Size is (C, T, H, W)

        Returns:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes resized
        """
        image, annot, meta = volume
        assert image.shape[1:] == annot.shape[1:]
        size = self.size
        if self.scale_factor:
            size = (torch.tensor(image.shape[1:], dtype=torch.float) * self.scale_factor).floor().int().tolist()
        image, annot = (
            F.resize(image, size, self.interpolation),
            F.resize(annot, size, 'nearest'),
        )
        meta['size'] = tuple(image.shape[1:])

        return image, annot, meta

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


# noinspection PyTypeChecker
class CenterCropImageVolume(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of image volume. Size is
            (C, T, crop_size, crop_size)
        """
        image, annot, meta = volume
        return (
            F.center_crop(image, self.crop_size),
            F.center_crop(annot, self.crop_size),
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={0})'.format(self.crop_size)


class NormalizeImageVolume(object):
    """
    Normalize the image volume by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        raise NotImplementedError
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        """
        image, annot, meta = volume
        return (
            F.normalize(image, self.mean, self.std, self.inplace),
            annot,
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(
            self.mean, self.std, self.inplace)


class ToTensorImageVolume(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of volume tensor
    """

    def __init__(self):
        pass

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Return:
            volume (tuple(torch.tensor, torch.tensor, dict)): Output image and mask volumes. Size is (C, T, H, W)
        """
        image, annot, meta = volume
        return (
            F.to_tensor(image),
            annot,
            meta
        )

    def __repr__(self):
        return self.__class__.__name__


class RandomFlipImageVolume(object):
    """
    Flip the image volume along the given direction with a given probability
    Args:
        p (float): probability of the volume being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, dim=3):
        self.p = p
        self.dim = dim

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Return:
            volume (tuple(torch.tensor, torch.tensor, dict)): Output image and mask volumes. Size is (C, T, H, W)
        """
        image, annot, meta = volume
        dim = self.dim
        if self.dim < 0:
            dim = random.randint(0, 2)
        if random.random() < self.p:
            image = F.flip(image, dim)
            annot = F.flip(annot, dim)
        return (
            image,
            annot,
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
