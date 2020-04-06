"""
Implemented by Mahdi Biparva, April 2020 @ Sunnybrook Research Institure (SRI) - Brain Imaging Lab (BIL)
Inspired from the torchvision transform files for video processing.
This contains a list of transformation modules to process image volumes.
Image volumes represent dense 3D volumes generated from CT/MRI scans.
"""

import numbers
import random

from torchvision.transforms import (
    RandomCrop,
    RandomResizedCrop,
)

from . import functional_mitorch as F


__all__ = [
    "RandomCropImageVolume",
    "RandomResizedCropImageVolume",
    "CenterCropImageVolume",
    "NormalizeImageVolume",
    "ToTensorImageVolume",
    "RandomHorizontalFlipImageVolume",
]


# noinspection PyMissingConstructor
class RandomCropImageVolume(RandomCrop):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, OH, OW)
        """
        image, annot = volume
        i, j, h, w = self.get_params(volume, self.size)
        return (
            F.crop(image, i, j, h, w),
            F.crop(annot, i, j, h, w)
        )

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


# noinspection PyMissingConstructor
class RandomResizedCropImageVolume(RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, H, W)
        """
        image, annot = volume
        i, j, h, w = self.get_params(volume, self.scale, self.ratio)
        return (
            F.resized_crop(image, i, j, h, w, self.size, self.interpolation_mode),
            F.resized_crop(annot, i, j, h, w, self.size, 'nearest')
        )

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2}, ratio={3})'.format(
                self.size, self.interpolation_mode, self.scale, self.ratio
            )


class CenterCropImageVolume(object):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of image volume. Size is
            (C, T, crop_size, crop_size)
        """
        image, annot = volume
        return (
            F.center_crop(image, self.crop_size),
            F.center_crop(annot, self.crop_size)
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
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        """
        image, annot = volume
        return (
            F.normalize(image, self.mean, self.std, self.inplace),
            annot
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
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Return:
            volume (tuple(torch.tensor, torch.tensor)): Output image and mask volumes. Size is (C, T, H, W)
        """
        image, annot = volume
        return (
            F.to_tensor(image),
            annot
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
            volume (tuple(torch.tensor, torch.tensor)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Return:
            volume (tuple(torch.tensor, torch.tensor)): Output image and mask volumes. Size is (C, T, H, W)
        """
        image, annot = volume
        if random.random() < self.p:
            image = F.flip(image, self.dim)
            annot = F.flip(annot, self.dim)
        return (
            image,
            annot
        )

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)
