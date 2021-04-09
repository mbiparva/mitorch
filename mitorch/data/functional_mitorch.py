#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import sys
import torch
import numbers
import collections
import numpy as np
import utils.k_space_motion as ks_motion
from data.utils_ext import _is_tensor_image_volume


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


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
        min_side (int): does it use minimum or maximum side if target_size
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
    return volume.permute(3, 0, 1, 2)


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
    maximum = volume_reshaped.max(1)[0] - minimum
    maximum[maximum < 1.0] = 1.0
    if max_div and not (maximum == 1.0).all():
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


# For more information check: https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/exposure.py
def scale_tensor_intensity(volume, input_range, output_range):
    def within_range(x):
        return volume.min().item() <= x <= volume.max().item()
    assert isinstance(volume, torch.Tensor), 'only accept torch tensors'
    assert isinstance(input_range, (tuple, list)), 'input_range must be either tuple or list'
    assert isinstance(output_range, (tuple, list)), 'output_range must be either tuple or list'
    assert len(input_range) == 2, 'len of input_range must be two'
    assert len(output_range) == 2, 'len of output_range must be two'
    assert all(map(within_range, input_range)), 'input_range values must be in [0, 1]'
    assert all(map(within_range, output_range)), 'output_range values must be in [0, 1]'
    in_lower, in_upper = tuple(map(float, input_range))
    out_lower, out_upper = tuple(map(float, output_range))

    volume = volume.clamp(in_lower, in_upper)

    if in_lower == in_upper:
        return volume.clamp(out_lower, out_upper)
    else:
        volume = (volume - in_lower) / (in_upper - in_lower)
        return volume * (out_upper - out_lower) + out_lower


def gamma_correction(volume, gamma):
    assert isinstance(gamma, float), 'gamma must be float'
    assert 0 < gamma, 'gamma must be greater than zero'

    in_lower, in_upper = volume.min().item(), volume.max().item()
    vol_range = in_upper - in_lower

    volume = (volume - in_lower) / vol_range

    volume = volume ** gamma

    return volume * vol_range + in_lower


def log_correction(volume, inverse):
    assert isinstance(inverse, bool), 'inverse must be bool'

    in_lower, in_upper = volume.min().item(), volume.max().item()
    vol_range = in_upper - in_lower

    volume = (volume - in_lower) / vol_range

    if inverse:
        volume = (2 ** volume - 1)
    else:
        volume = np.log2(1 + volume)

    return volume * vol_range + in_lower


def sigmoid_correction(volume, inverse, gain, cutoff):
    assert isinstance(inverse, bool), 'inverse must be bool'

    in_lower, in_upper = volume.min().item(), volume.max().item()
    vol_range = in_upper - in_lower

    volume = (volume - in_lower) / vol_range

    if inverse:
        volume = 1 - 1 / (1 + np.exp(gain * (cutoff - volume)))
    else:
        volume = 1 / (1 + np.exp(gain * (cutoff - volume)))

    return volume * vol_range + in_lower


def histogram(volume, num_bins=256, is_normalized=False):
    hist, bin_edges = np.histogram(volume.numpy().flatten(), bins=num_bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if is_normalized:
        hist = hist / np.sum(hist)
    return hist, bin_centers


def cumulative_distribution(volume, num_bins=256):
    hist, bin_centers = histogram(volume, num_bins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / img_cdf[-1]
    return img_cdf, bin_centers


def equalize_hist(volume, num_bins=256):
    in_lower, in_upper = volume.min().item(), volume.max().item()
    vol_range = in_upper - in_lower

    cdf, bin_centers = cumulative_distribution(volume, num_bins)
    volume_int = np.interp(volume.numpy().flatten(), bin_centers, cdf)
    volume_int = volume_int.reshape(volume.shape)
    volume_int = torch.from_numpy(volume_int)

    return volume_int * vol_range + in_lower


# For more information check: https://github.com/dipy/dipy/blob/master/dipy/sims/voxel.py
def additive_noise(volume, sigma, noise_type='rician', out_of_bound_mode='normalize'):
    assert out_of_bound_mode in ('normalize', 'clamp',), 'undefined out_of_bound_mode'
    noise_function = {
        'gaussian': lambda x, n1, n2: x + n1,
        'rician': lambda x, n1, n2: np.sqrt((x + n1) ** 2 + n2 ** 2),
        'rayleigh': lambda x, n1, n2: x + np.sqrt(n1 ** 2 + n2 ** 2),
    }

    in_lower, in_upper = volume.min().item(), volume.max().item()
    vol_range = in_upper - in_lower

    volume = (volume - in_lower) / vol_range

    sigma = volume.std() * sigma
    noise_one = np.random.normal(0, sigma, size=volume.shape)
    noise_two = np.random.normal(0, sigma, size=volume.shape)

    volume = noise_function[noise_type](volume, noise_one, noise_two)

    if out_of_bound_mode == 'normalize':
        noise_in_lower, noise_in_upper = volume.min().item(), volume.max().item()
        noise_vol_range = noise_in_upper - noise_in_lower
        volume = (volume - noise_in_lower) / noise_vol_range
    elif out_of_bound_mode == 'clamp':
        volume = volume.clamp(0, 1)
    else:
        raise NotImplementedError

    volume = volume * vol_range + in_lower

    return volume


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype, dim: int,
            ignore_background=True):
    """
    This coverts a categorical annotation tensor to one-hot annotation tensor.
    It is adapted from MONAI at the link below:

    Reference:
    https://github.com/Project-MONAI/MONAI/blob/09f39dcb84092b07cda480c99644f9f7f8cceab6/monai/networks/utils.py#L24

    Args:
        labels: the input label tensor to convert
        num_classes: number of classes
        dtype: dtype to return the output tensor
        dim: where to put the new dimension for labels
        ignore_background: drops the first sheet for background. Assumes the first index is background.

    Returns: one-hot tensor
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    shape_tensor = list(labels.shape)

    assert shape_tensor[dim] == 1, "labels should have a channel with length equals to one."
    shape_tensor[dim] = num_classes

    labels_one_hot = torch.zeros(size=shape_tensor, dtype=dtype, device=labels.device)
    labels = labels_one_hot.scatter_(dim=dim, index=labels.long(), value=1)

    if ignore_background:
        keep_ind = torch.tensor(range(1, num_classes))  # always assumes index 0 is background
        labels = labels.index_select(dim=dim, index=keep_ind)

    return labels
<<<<<<< HEAD
=======


def k_space_motion_artifact(volume, time, **kwargs):
    """
    Args:
        volume (torch.Tensor): Volume to be transformed and resampled. Must be 4D
            with a channel dimension i.e. (C, D, H, W).
        time (float): Time at which the motion occurs during scanning. Should be between [0.5, 1), where 0
            represents the beginning of the scan and 1 represents the end. Time >= 0.5 assures that the
            most prominent object in the image is in the original position of the image so that ground truth
            annotations don't need to be adjusted.
    Returns:
        volume (torch.Tensor): Motion-artifacted image. Shape is the same as the input.
    """
    assert _is_tensor_image_volume(volume), 'volume should be a 4D torch.Tensor with a channel dimension'
    assert isinstance(time, float), 'time must be float between 0.0 (inclusive) and 1.0 (exclusive).'
    assert 0.5 <= time < 1.0, 'time must be float between 0.0 (inclusive) and 1.0 (exclusive).'

    return ks_motion.apply_motion_from_affine_params(volume, time, **kwargs)
>>>>>>> 376197d... motion: I merely refactored and made some style fixes and relocation
