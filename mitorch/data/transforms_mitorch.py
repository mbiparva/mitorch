#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)


import torch
import numbers
import random
from . import functional_mitorch as F
import collections
import sys
import nibabel as nib
from data.VolSet import c3d_labels
import numpy as np
from data.ABC_utils import Transformable, Randomizable
import itertools

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


__all__ = [
    'RandomOrientationTo',
    'RandomResampleTomm',
    'RandomCropImageVolume',
    'RandomResizedCropImageVolume',
    'ResizeImageVolume',
    'CenterCropImageVolume',
    'NormalizeMeanStdVolume',
    'NormalizeMinMaxVolume',
    'ToTensorImageVolume',
    'RandomFlipImageVolume',
    'PadVolume',
    'PadToSizeVolume',
]


def generate_all_possible_orients(labels_set):
    return list(map(
        lambda x: ''.join(x),
        itertools.chain(*map(
            lambda x: itertools.product(*x),
            itertools.permutations(c3d_labels)
        ))
    ))


# noinspection PyTypeChecker
class RandomOrientationTo(Randomizable):
    def __init__(self, target_orient, orientation_set=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c3d_labels_str = ''.join([''.join(i) for i in c3d_labels])
        assert isinstance(target_orient, str), 'target_orient must be string'
        assert all([i in c3d_labels_str for i in target_orient]), 'letters in target_orient must be in {}'.format(
            c3d_labels_str
        )
        if orientation_set:
            assert isinstance(orientation_set, Iterable)
        self.target_orient = target_orient.upper()
        if orientation_set is None:
            self.orientation_set = generate_all_possible_orients(c3d_labels)

    def randomize_params(self, volume):
        self.target_orient = self.orientation_set[
            torch.randint(
                len(self.orientation_set),
                (1,)
            )
        ]

    @staticmethod
    def apply_orient(tensor, orient_trans):
        return torch.from_numpy(
            np.ascontiguousarray(
                nib.apply_orientation(tensor, orient_trans)
            )
        )

    def apply(self, volume):
        image, annot, meta = volume
        affine = torch.tensor(meta['affine'], dtype=torch.float)
        affine = affine.reshape(4, 4)

        orient_source = nib.io_orientation(affine)
        orient_final = nib.orientations.axcodes2ornt(self.target_orient, labels=c3d_labels)
        orient_trans = nib.orientations.ornt_transform(orient_source, orient_final)

        orient_trans_3d = orient_trans.copy()
        orient_trans[:, 0] += 1  # we skip the channel dimension
        orient_trans = np.concatenate([np.array([[0, 1]]), orient_trans])

        orient_identical = np.array(list(zip(range(4), [1]*4)))
        if not np.all(orient_trans == orient_identical):  # identical rotation
            image = self.apply_orient(image, orient_trans)
            annot = self.apply_orient(annot, orient_trans)

            image_shape = image.shape[1:]
            inv_affine_trans = nib.orientations.inv_ornt_aff(orient_trans_3d, image_shape)
            affine = affine.mm(torch.from_numpy(inv_affine_trans).float())
            meta['affine'] = affine.flatten().tolist()

            # update size and spacing using affine and shape
            meta['size'] = tuple(image.shape[1:])
            dim = meta["dimension"]
            RZS = affine[:dim, :dim].numpy()
            zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
            zooms[zooms == 0] = 1
            meta['spacing'] = zooms
            meta["direction"] = list((RZS / zooms).flatten())

        return (
            image,
            annot,
            meta
        )


class RandomResampleTomm(Randomizable):
    def __init__(self, target_spacing=(1, 1, 1), target_spacing_scale=(0.2, 0.2, 0.2),
                 interpolation='trilinear', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpolation = interpolation
        assert isinstance(target_spacing, (tuple, list)) and len(target_spacing) == 3
        assert isinstance(target_spacing_scale, (tuple, list)), len(target_spacing_scale) == 3
        self.target_spacing = self.target_spacing_constant = torch.tensor(target_spacing, dtype=torch.float32)
        self.target_spacing_scale = torch.tensor(target_spacing_scale, dtype=torch.float32)

    def randomize_params(self, volume):
        self.target_spacing = ((torch.rand(3) - 1/2) * 2 * self.target_spacing_scale + 1) * self.target_spacing_constant

    def apply(self, volume):
        image, annot, meta = volume
        size = torch.tensor(meta['size'], dtype=torch.float)
        spacing = torch.tensor(meta['spacing'], dtype=torch.float)
        spacing /= self.target_spacing
        iso1mm = torch.tensor((1, 1, 1), dtype=torch.float32)
        if (spacing == iso1mm).all().item() or torch.allclose(spacing, iso1mm, rtol=1e-3, atol=0):
            return volume
        # size = (size * spacing).floor().int().tolist()[::-1]  # reverse size since F.resize works in DxHxW space
        size = (size * spacing).floor().int().tolist()
        image, annot = (
            F.resize(image, size, self.interpolation),
            F.resize(annot, size, 'nearest'),
        )
        # meta['size'] = size[::-1]
        meta['size'] = size
        meta['spacing'] = self.target_spacing.tolist()
        dim = meta["dimension"]
        affine_ = torch.tensor(meta['affine'], dtype=torch.float)
        affine_ = affine_.reshape(4, 4)
        affine = affine_[:dim, :dim]
        # affine[affine != 0] = torch.tensor(meta['spacing'])  # this assumes affine has not rotation just scaling
        affine[range(len(affine)), range(len(affine))] = torch.tensor(meta['spacing']) * affine.diagonal().sign()
        meta['affine'] = affine_.flatten().tolist()

        return image, annot, meta


# noinspection PyMissingConstructor,PyTypeChecker
class RandomCropImageVolume(Randomizable):
    def __init__(self, size, *args, **kwargs):
        if 'prand' in kwargs and not kwargs['prand']:
            raise ValueError('If you want to turn prand off, use CenterCropImageVolume instead. '
                             'This one does crop location randomization by default')
        kwargs['prand'] = True  # we always set prand to True for ths particular transform
        super().__init__(*args, **kwargs)
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size
        assert self.prand, 'If you want to turn prand off, use CenterCropImageVolume instead.' \
                           'This one does crop location randomization by default'
        self.rloc = None

    def randomize_params(self, volume):
        image_shape = volume[0].shape  # image, annot, meta
        self.rloc = self.get_params(image_shape, self.size)

    @staticmethod
    def get_params(volume_shape, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            volume_shape (Torch Tensor): Shape of the volume to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        d, h, w = volume_shape[1:]
        td, th, tw = output_size
        if d == td and h == th and w == tw:
            return 0, 0, 0, d, h, w

        k = random.randint(0, d - td)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return k, i, j, td, th, tw

    def apply(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image, mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, OH, OW)
        """
        image, annot, meta = volume
        k, i, j, d, h, w = self.rloc
        return (
            F.crop(image, k, i, j, d, h, w),
            F.crop(annot, k, i, j, d, h, w),
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


# noinspection PyMissingConstructor
class RandomResizedCropImageVolume(Randomizable):
    def __init__(self, size, scale=(0.80, 1.0), interpolation='trilinear', uni_scale=True, *args, **kwargs):
        if 'prand' in kwargs and not kwargs['prand']:
            raise ValueError('If you want to turn prand off, use CenterCropImageVolume instead. '
                             'This one does crop location randomization by default')
        kwargs['prand'] = True  # we always set prand to True for ths particular transform
        super().__init__(*args, **kwargs)
        assert isinstance(scale, tuple) and len(scale) == 2, 'scale is not defined right'
        assert 0 < scale[0] < scale[1] <= 1, 'scale must fall in (lower_range, upper_range)'
        assert isinstance(uni_scale, bool), 'iso_crop is bool'
        if isinstance(size, tuple):
            assert len(size) == 3, "size should be tuple (depth, height, width)"
            self.size = size
        else:
            self.size = tuple([int(size)]*3)

        self.interpolation = interpolation
        self.scale = scale
        self.uni_scale = uni_scale
        self.rloc = None

    def randomize_params(self, volume):
        image_shape = volume[0].shape  # image, annot, meta
        self.rloc = self.get_params(image_shape, self.scale, self.uni_scale)

    @staticmethod
    def get_params(volume_shape, scale, uni_scale):
        """Get parameters for ``crop`` for a random crop.

        Args:
            volume_shape (Torch Tensor): Shape of the volume to be cropped.
            scale (tuple): Expected output size of the crop.
            uni_scale: uniformly scale all three sides

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        d, h, w = volume_shape[1:]

        if uni_scale:
            scale_rnd = random.uniform(*scale)
            td, th, tw = (torch.tensor(volume_shape[1:]) * scale_rnd).round().int().tolist()
        else:
            td_l, th_l, tw_l = (torch.tensor(volume_shape[1:]) * scale[0]).round().int().tolist()
            td_u, th_u, tw_u = (torch.tensor(volume_shape[1:]) * scale[1]).round().int().tolist()
            td = random.randint(td_l, td_u)
            th = random.randint(th_l, th_u)
            tw = random.randint(tw_l, tw_u)

        k = random.randint(0, d - td)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return k, i, j, td, th, tw

    def apply(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized image volume.
                size is (C, T, H, W)
        """
        image, annot, meta = volume
        k, i, j, d, h, w = self.rloc
        return (
            F.resized_crop(image, k, i, j, d, h, w, self.size, self.interpolation),
            F.resized_crop(annot, k, i, j, d, h, w, self.size, 'nearest'),
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + \
            '(size={0}, interpolation_mode={1}, scale={2})'.format(
                self.size, self.interpolation, self.scale
            )


# noinspection PyTypeChecker
class ResizeImageVolume(Transformable):
    """Resize the input image volume to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (d, h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if depth > height > width, then image will be rescaled to
            (size * depth / width, size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is trilinear
    """

    def __init__(self, size=None, scale_factor=None, interpolation='trilinear', min_side=True, ignore_depth=False):
        assert size or scale_factor, 'either size or scale_factor must be given'
        assert isinstance(min_side, bool)
        assert isinstance(ignore_depth, bool)
        if size:
            assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 3)
        if scale_factor:
            assert isinstance(scale_factor, float)
        if isinstance(size, Iterable) and len(size) == 3 and ignore_depth:
            print('warning: ignore_depth is valid when target_size is int')
        self.scale_factor = scale_factor
        self.size = size
        self.interpolation = interpolation
        self.min_side = min_side
        self.ignore_depth = ignore_depth

    def apply(self, volume):
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
            F.resize(image, size, self.interpolation, self.min_side, self.ignore_depth),
            F.resize(annot, size, 'nearest', self.min_side, self.ignore_depth),
        )
        meta['size'] = tuple(image.shape[1:])

        return image, annot, meta

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


# noinspection PyTypeChecker
class CenterCropImageVolume(Transformable):
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = tuple([int(crop_size)]*3)
        else:
            self.crop_size = crop_size

    def apply(self, volume):
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


class NormalizeMeanStdVolume(Transformable):
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

    def apply(self, volume):
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


class NormalizeMinMaxVolume(Transformable):
    """
    Normalize the image volume by minimum subtraction and division by maximum
    Args:
        max_div (3-tuple): divide by the maximum
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, max_div=True, inplace=False):
        self.max_div = max_div
        self.inplace = inplace

    def apply(self, volume):
        """
        Args:
            volume (tuple(torch.tensor, torch.tensor, dict)): Image and mask volumes to be cropped. Size is (C, T, H, W)
        """
        image, annot, meta = volume
        return (
            F.normalize_minmax(image, self.max_div, self.inplace),
            annot,
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(max_div={0}, inplace={1})'.format(self.max_div, self.inplace)


class ToTensorImageVolume(Transformable):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of volume tensor
    """

    def apply(self, volume):
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


class RandomFlipImageVolume(Transformable):
    """
    Flip the image volume along the given direction with a given probability
    Args:
        p (float): probability of the volume being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, dim=3):
        self.p = p
        self.dim = dim

    def apply(self, volume):
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


class PadVolume(Transformable):
    """Pad the given Torch Tensor Volume on all sides with the given "pad" value.

    Args:
        padding (Number or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right, 4 left/right and top/bottom, and 6 left/right, top/bottom, and front/back respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length K, it is used to fill all of the K channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: 'constant', 'reflect', 'replicate' or 'circular'. Default is constant.
            check torch.nn.functional.pad for further details
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'reflect', 'replicate', 'circular']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4, 6]:
            raise ValueError("Padding must be an int or a 2, 4, or 6 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def apply(self, volume):
        """
        Args:
            volume (Torch Tensor): Volume to be padded.

        Returns:
            Torch Tensor: Padded volume.
        """
        image, annot, meta = volume
        return (
            F.pad(image, self.padding, self.fill, self.padding_mode),
            F.pad(annot, self.padding, 0, self.padding_mode),  # TODO assumes bg is always zero, change it
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class PadToSizeVolume(Transformable):
    """Pad the given Torch Tensor Volume on all sides to have the given size.

    Args:
        target_size (Number or tuple): Target size to be padded to. If a single int is provided this
            is used to pad all borders. Otherwise, a tuple of length 3 is needed to se the target size of the volume.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length K, it is used to fill all of the K channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: 'constant', 'reflect', 'replicate' or 'circular',
        'mean', 'median', 'min', 'max'. Default is constant.
            check torch.nn.functional.pad for further details
    """

    def __init__(self, target_size, fill=0, padding_mode='constant'):
        assert isinstance(target_size, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'reflect', 'replicate', 'circular',
                                'mean', 'median', 'min', 'max'
                                ]
        if isinstance(target_size, Sequence) and not len(target_size) == 3:
            raise ValueError("Size must be an int or a 3 element tuple, not a " +
                             "{} element tuple".format(len(target_size)))

        if isinstance(target_size, numbers.Number):
            target_size = tuple([target_size]*3)
        self.target_size = torch.tensor(target_size)

        if isinstance(target_size, Sequence) and (self.target_size == -1).all():
            raise ValueError("all of the target size cannot set to auto_fill (-1). "
                             "Maximum must be < 3.")

        self.fill = fill
        self.padding_mode = padding_mode

    def apply(self, volume):
        """
        Args:
            volume (Torch Tensor): Volume to be padded.

        Returns:
            Torch Tensor: Padded volume.
        """
        target_size = self.target_size.clone()
        auto_fill_ind = target_size == -1
        image, annot, meta = volume
        image_size = torch.tensor(image.shape[1:])  # index 0 is the channel
        target_size[auto_fill_ind] = image_size[auto_fill_ind]
        assert (image_size <= target_size).all()
        size_offset = target_size - image_size
        padding_before = size_offset // 2
        padding_after = size_offset - padding_before
        padding = tuple(torch.stack((padding_before.flip(0), padding_after.flip(0))).T.flatten().tolist())

        fill = self.fill
        padding_mode = self.padding_mode
        if self.padding_mode in ('mean', 'median', 'min', 'max'):
            fill = getattr(image, self.padding_mode)().item()
            padding_mode = 'constant'

        return (
            F.pad(image, padding, fill, padding_mode),
            F.pad(annot, padding, 0, padding_mode),  # TODO assumes bg is always zero, change it
            meta
        )

    def __repr__(self):
        return self.__class__.__name__ + '(target_size={0}, fill={1}, padding_mode={2})'.\
            format(self.target_size, self.fill, self.padding_mode)


class RandomBrightness(Randomizable):
    def __init__(self, value, channel_wise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(value, float), 'value must be float'
        assert -0.5 <= value <= +0.5, 'value must be between [-0.5, +0.5]'
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise
        self.value = value

    def randomize_params(self, volume):
        self.value = torch.rand(1).item() - 1/2

    def update_value(self, image):
        img_min, img_max = image.min().item(), image.max().item()
        img_range = img_max - img_min
        self.value = img_range * self.value

    def find_ranges(self, image):
        img_min, img_max = image.min().item(), image.max().item()
        if self.value > 0:
            input_range, output_range = (img_min, img_max), (img_min + self.value, img_max)
        else:
            input_range, output_range = (img_min, img_max), (img_min, img_max + self.value)

        return input_range, output_range

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            self.update_value(image_i)

            if not self.value == 0:
                input_range, output_range = self.find_ranges(image_i)

                image_i = F.scale_tensor_intensity(image_i, input_range, output_range)

            if self.channel_wise:
                image[i] = image_i
                self.randomize(image)
            else:
                image = image_i
                break

        return image, annot, meta


class RandomContrast(RandomBrightness):
    def __init__(self, value, channel_wise=True, *args, **kwargs):
        super().__init__(value, channel_wise, *args, **kwargs)

    def find_ranges(self, image):
        img_min, img_max = image.min().item(), image.max().item()
        if self.value < 0:
            input_range, output_range = (img_min, img_max), (img_min - self.value, img_max + self.value)
        else:
            input_range, output_range = (img_min + self.value, img_max - self.value), (img_min, img_max)

        return input_range, output_range


class RandomGamma(Randomizable):
    POWER_MAX = 6

    def __init__(self, value, channel_wise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(value, float), 'value must be float'
        assert 0 < value, 'value must be greater than zero'
        assert value < self.POWER_MAX, 'large value greater than 6 are not recommended for numerical stability'
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise
        self.value = value

    def randomize_params(self, volume):
        # to be equal we pick randomly under or over contrast
        value_under = torch.rand(1).item()
        value_over = torch.rand(1).item() * self.POWER_MAX
        self.value = random.choice((value_under, value_over))

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            image_i = F.gamma_correction(image_i, self.value)

            if self.channel_wise:
                image[i] = image_i
                self.randomize(image)
            else:
                image = image_i
                break

        return image, annot, meta


class LogCorrection(Transformable):  # TODO might be interesting to randomize inverse
    def __init__(self, inverse=False, channel_wise=True):
        assert isinstance(inverse, bool), 'inverse must be bool'
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise
        self.inverse = inverse

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            image_i = F.log_correction(image_i, self.inverse)

            if self.channel_wise:
                image[i] = image_i
            else:
                image = image_i
                break

        return image, annot, meta


class SigmoidCorrection(Transformable):  # TODO might be interesting to randomize inverse
    def __init__(self, inverse=False, gain=10, cutoff=0.5, channel_wise=True):
        assert isinstance(inverse, bool), 'inverse must be bool'
        assert 0 < cutoff <= 1, 'cutoff is between [0, 1]'
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise
        self.inverse = inverse
        self.gain = gain
        self.cutoff = cutoff

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            image_i = F.sigmoid_correction(image_i, self.inverse, self.gain, self.cutoff)

            if self.channel_wise:
                image[i] = image_i
            else:
                image = image_i
                break

        return image, annot, meta


class HistEqual(Transformable):
    def __init__(self, num_bins=256, channel_wise=True):
        assert isinstance(num_bins, int), 'num_bins must be int'
        assert 0 < num_bins
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise
        self.num_bins = num_bins

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            image_i = F.equalize_hist(image_i, self.num_bins)

            if self.channel_wise:
                image[i] = image_i
            else:
                image = image_i
                break

        return image, annot, meta


class AdditiveNoise(Randomizable):
    NOISE_TYPE = ('gaussian', 'rician', 'rayleigh',)
    MAX_SIGMA = 1.5

    def __init__(self, sigma, noise_type='gaussian', randomize_type=False, out_of_bound_mode='normalize',
                 channel_wise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(sigma, float), 'sigma must be float'
        assert 0 < sigma, 'sigma must be greater than zero'
        assert noise_type in self.NOISE_TYPE, 'unknown noise type'
        assert out_of_bound_mode in ('normalize', 'clamp',), 'undefined out_of_bound_mode'
        assert isinstance(channel_wise, bool), 'channel_wise is bool'
        self.channel_wise = channel_wise

        self.sigma = sigma
        self.noise_type = noise_type
        self.randomize_type = randomize_type
        self.out_of_bound_mode = out_of_bound_mode

    def randomize_params(self, volume):
        self.sigma = torch.rand(1).item() * self.MAX_SIGMA
        if self.randomize_type:
            self.noise_type = random.choice(self.NOISE_TYPE)

    def apply(self, volume):
        image, annot, meta = volume

        for i in range(len(image)):
            if self.channel_wise:
                image_i = image[i]
            else:
                image_i = image

            image_i = F.additive_noise(image_i, self.sigma, self.noise_type, self.out_of_bound_mode)

            if self.channel_wise:
                image[i] = image_i
                self.randomize(image)
            else:
                image = image_i
                break

        return image, annot, meta


# TODO Implement CropTightVolume based off of
#  https://github.com/nilearn/nilearn/blob/c10248e43769f37eaea804f64d44a7816e3c6e03/nilearn/image/image.py
