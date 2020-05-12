#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import sys
import torch
import numbers
import collections
import numpy as np

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_tensor_image_volume(volume):
    if not torch.is_tensor(volume):
        raise TypeError("volume should be Tensor. Got %s" % type(volume))

    if not volume.dim() == 4:
        raise ValueError("volume should be 4D. Got %dD" % volume.dim())

    return True


def crop(volume, k, i, j, d, h, w):
    """
    Args:
        volume (torch.tensor): Image volume to be cropped. Size is (C, T, H, W)
        k (int): k in (k,i,j) i.e coordinates of the back upper left corner.
        i (int): i in (k,i,j) i.e coordinates of the back upper left corner.
        j (int): j in (k,i,j) i.e coordinates of the back upper left corner.
        d (int): Depth of the cropped region.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
    """
    assert _is_tensor_image_volume(volume)
    return volume[..., k:k + d, i:i + h, j:j + w]


def resize(volume, target_size, interpolation_mode, min_side=True, ignore_depth=False):
    r"""
    Resize the image volume using the target size and interpolation mode.
    It uses the torch.nn.functional.interpolate function.

    Args:
        volume (torch.tensor): the image volume
        target_size (Tuple[int, int, int]): the target size
        min_side (int): does it use minimum or maximum side if target_size is int
        interpolation_mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        ignore_depth (bool): Ignore resizing in the depth dimension when size is int
    Returns:
        volume (torch.tensor): Resized volume. Size is (C, T, H, W)

    """
    assert isinstance(target_size, int) or len(target_size) == 3, "target size must be int or " \
                                                                  "tuple (depth, height, width)"
    assert isinstance(min_side, bool), "min_size must be bool"
    assert isinstance(ignore_depth, bool), "ignore_depth is bool"
    if isinstance(target_size, Sequence) and len(target_size) == 3 and ignore_depth:
        print('warning: ignore_depth is valid when target_size is int')
    if isinstance(target_size, int):
        _, d, h, w = volume.shape
        dim_min = min(d, h, w) if min_side else max(d, h, w)
        if dim_min == target_size:
            return volume
        if dim_min == w:
            ow = target_size
            oh = int(target_size * h / w)
            od = int(target_size * d / w) if not ignore_depth else d
        elif dim_min == h:
            oh = target_size
            ow = int(target_size * w / h)
            od = int(target_size * d / h) if not ignore_depth else d
        else:
            od = target_size
            ow = int(target_size * w / d)
            oh = int(target_size * h / d)
        target_size = (od, oh, ow)
    if interpolation_mode == 'nearest':
        return torch.nn.functional.interpolate(
            volume.unsqueeze(dim=0),
            size=target_size,
            mode=interpolation_mode,
        ).squeeze(dim=0)
    else:
        return torch.nn.functional.interpolate(
            volume.unsqueeze(dim=0),
            size=target_size,
            mode=interpolation_mode,
            align_corners=False
        ).squeeze(dim=0)


def resized_crop(volume, k, i, j, d, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the image volume
    Args:
        volume (torch.tensor): Image volume to be cropped. Size is (C, T, H, W)
        k (int): k in (k,i,j) i.e coordinates of the back upper left corner.
        i (int): i in (k,i,j) i.e coordinates of the back upper left corner.
        j (int): j in (k,i,j) i.e coordinates of the back upper left corner.
        d (int): Depth of the cropped region.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized volume
        interpolation_mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    Returns:
        volume (torch.tensor): Resized and cropped volume. Size is (C, T, H, W)
    """
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    volume = crop(volume, k, i, j, d, h, w)
    volume = resize(volume, size, interpolation_mode)
    return volume


# noinspection PyTypeChecker
def center_crop(volume, crop_size):
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    if not isinstance(crop_size, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate crop_size arg')
    if isinstance(crop_size, Sequence) and not len(crop_size) == 3:
        raise ValueError("crop_size must be an int or 3 element tuple, not a " +
                         "{} element tuple".format(len(crop_size)))
    if isinstance(crop_size, numbers.Number):
        crop_size = tuple([int(crop_size)]*3)
    d, h, w = volume.shape[1:]
    td, th, tw = crop_size
    assert d >= td and h >= th and w >= tw, "depth, height and width must not be smaller than crop_size"

    k = int(round((d - td) / 2.0))
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(volume, k, i, j, td, th, tw)


def to_tensor(volume):
    """
    Convert tensor data type rto float and permute the dimenions of volume tensor
    Args:
        volume (torch.tensor, dtype=torch.int): Size is (T, H, W, C)
    Return:
        volume (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    _is_tensor_image_volume(volume)
    if not volume.dtype == torch.float32:
        raise TypeError("volume tensor should have data type torch.float32. Got %s" % str(volume.dtype))
    return volume.permute(3, 0, 1, 2) / 1  # TODO: decide whether division is needed


def normalize(volume, mean, std, inplace=False):
    """
    Args:
        volume (torch.tensor): Image volume to be normalized. Size is (C, T, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
        inplace (bool): inplace operation
    Returns:
        normalized volume (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    if not inplace:
        volume = volume.clone()
    mean = torch.as_tensor(mean, dtype=volume.dtype, device=volume.device)
    std = torch.as_tensor(std, dtype=volume.dtype, device=volume.device)
    volume.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return volume


def normalize_minmax(volume, max_div, inplace=False):
    """
    Args:
        volume (torch.tensor): Image volume to be normalized. Size is (C, T, H, W)
        max_div (bool): whether divide by the maximum (max of one).
        inplace (bool): inplace operation
    Returns:
        normalized volume (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    if not inplace:
        volume = volume.clone()
    volume_reshaped = volume.reshape(volume.size(0), -1)
    minimum = volume_reshaped.min(1)[0]
    volume.sub_(minimum[:, None, None, None])
    if max_div:
        maximum = volume_reshaped.max(1)[0]
        volume.div_(maximum[:, None, None, None])

    return volume


def flip(volume, dim=3):
    """
    Args:
        volume (torch.tensor): Image volume to be normalized. Size is (C, T, H, W)
        dim (int): the axis to flip the volume over it.
    Returns:
        flipped volume (torch.tensor): Size is (C, T, H, W)
    """
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    return volume.flip(dim)


def pad(volume, padding, fill=0, padding_mode='constant'):
    r"""Pad the given Tensor volume on all sides with specified padding mode and fill value.

    Args:
        volume (Torch Tensor): Volume to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: 'constant', 'reflect', 'replicate' or 'circular'. Default is constant.
            check torch.nn.functional.pad for further details ### Deprecated - check np.pad

    Returns:
        Torch Tensor: Padded volume.
    """
    _is_tensor_image_volume(volume)

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, Sequence) and len(padding) not in [2, 4, 6]:
        raise ValueError("Padding must be an int or a 2, 4, or 6 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'reflect', 'replicate', 'circular'], \
        'Padding mode should be either constant, reflect, replicate or circular'
    if isinstance(padding, int):
        padding = [padding]*6

    return torch.nn.functional.pad(volume, padding, mode=padding_mode, value=fill)
