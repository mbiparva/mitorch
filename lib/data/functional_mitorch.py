"""
Implemented by Mahdi Biparva, April 2020 @ Sunnybrook Research Institure (SRI) - Brain Imaging Lab (BIL)
Inspired from torchvision transform files for video processing.
This contains a list of transformation functionals to process image volumes.
Image volumes represent dense 3D volumes generated from CT/MRI scans.
"""

import torch


def _is_tensor_image_volume(volume):
    if not torch.is_tensor(volume):
        raise TypeError("volume should be Tesnor. Got %s" % type(volume))

    if not volume.dim() == 4:
        raise ValueError("volume should be 4D. Got %dD" % volume.dim())

    return True


def crop(volume, i, j, h, w):
    """
    Args:
        volume (torch.tensor): Image volume to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
    """
    raise NotImplementedError
    assert _is_tensor_image_volume(volume)
    return volume[..., i:i + h, j:j + w]


def resize(volume, target_size, interpolation_mode):
    r"""
    Resize the image volume using the target size and interpolation mode.
    It uses the torch.nn.functional.interpolate function.

    Args:
        volume (torch.tensor): the image volume
        target_size (Tuple[int, int, int]): the target size
        interpolation_mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    Returns:
        volume (torch.tensor): Resized volume. Size is (C, T, H, W)

    """
    raise NotImplementedError
    assert isinstance(target_size, int) or len(target_size) == 3, "target size must be int or " \
                                                                  "tuple (depth, height, width)"
    if isinstance(target_size, int):
        _, d, h, w = volume.shape
        dim_min = min(d, h, w)
        if dim_min == target_size:
            return volume
        if dim_min == w:
            ow = target_size
            oh = int(target_size * h / w)
            od = int(target_size * d / w)
        elif dim_min == h:
            oh = target_size
            ow = int(target_size * w / h)
            od = int(target_size * d / h)
        else:
            od = target_size
            ow = int(target_size * w / d)
            oh = int(target_size * h / d)
        target_size = (od, oh, ow)
    return torch.nn.functional.interpolate(volume, size=target_size, mode=interpolation_mode)


def resized_crop(volume, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the image volume
    Args:
        volume (torch.tensor): Image volume to be cropped. Size is (C, T, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized volume
        interpolation_mode (str): algorithm used for upsampling:
        ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
        ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
    Returns:
        volume (torch.tensor): Resized and cropped volume. Size is (C, T, H, W)
    """
    raise NotImplementedError
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    volume = crop(volume, i, j, h, w)
    volume = resize(volume, size, interpolation_mode)
    return volume


def center_crop(volume, crop_size):
    raise NotImplementedError
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    h, w = volume.size(-2), volume.size(-1)
    th, tw = crop_size
    assert h >= th and w >= tw, "height and width must be no smaller than crop_size"

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(volume, i, j, th, tw)


def to_tensor(volume):
    """
    Convert tensor data type rto float and permute the dimenions of volume tensor
    Args:
        volume (torch.tensor, dtype=torch.int): Size is (T, H, W, C)
    Return:
        volume (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
    """
    raise NotImplementedError
    _is_tensor_image_volume(volume)
    if not volume.dtype == torch.int:
        raise TypeError("volume tensor should have data type int. Got %s" % str(volume.dtype))
    return volume.float().permute(3, 0, 1, 2) / 1  # TODO: decide whether division is needed


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
    raise NotImplementedError
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    if not inplace:
        volume = volume.clone()
    mean = torch.as_tensor(mean, dtype=volume.dtype, device=volume.device)
    std = torch.as_tensor(std, dtype=volume.dtype, device=volume.device)
    volume.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return volume


def flip(volume, dim=3):
    """
    Args:
        volume (torch.tensor): Image volume to be normalized. Size is (C, T, H, W)
        dim (int): the axis to flip the volume over it.
    Returns:
        flipped volume (torch.tensor): Size is (C, T, H, W)
    """
    raise NotImplementedError
    assert _is_tensor_image_volume(volume), "volume should be a 4D torch.tensor"
    return volume.flip(dim)

# TODO Develop SimpleITK Modules like Adaptive Histogram Equalization