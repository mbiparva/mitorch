#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""
****** NOTE: ALL THE CODE BELOW ARE TAKEN FROM MONAI WITH THE COPYRIGHT NOTE BELOW ******
                    https://github.com/Project-MONAI/MONAI
"""

# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from abc import ABC, abstractmethod
import math
import random
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union, Iterable, Any, Generator, Dict, cast
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from warnings import warn
from itertools import product, starmap
import torch.nn.functional as F


MAX_SEED = np.iinfo(np.uint32).max + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32
HAS_EXT = USE_COMPILED = False
IndexSelection = Union[Iterable[int], int]
"""IndexSelection

The IndexSelection type is used to for defining variables
that store a subset of indices to select items from a List or Array like objects.
The indices must be integers, and if a container of indices is specified, the
container must be iterable.
"""


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default


class NumpyPadMode(Enum):
    """
    See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    CONSTANT = "constant"
    EDGE = "edge"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"
    WRAP = "wrap"
    EMPTY = "empty"


class Method(Enum):
    """
    See also: :py:class:`monai.transforms.croppad.array.SpatialPad`
    """

    SYMMETRIC = "symmetric"
    END = "end"


def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if torch.is_tensor(obj):
        return int(obj.dim()) > 0  # a 0-d tensor is not iterable
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = (vals,)

    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    Examples::

        >>> ensure_tuple_rep(1, 3)
        (1, 1, 1)
        >>> ensure_tuple_rep(None, 3)
        (None, None, None)
        >>> ensure_tuple_rep('test', 3)
        ('test', 'test', 'test')
        >>> ensure_tuple_rep([1, 2, 3], 3)
        (1, 2, 3)
        >>> ensure_tuple_rep(range(3), 3)
        (0, 1, 2)
        >>> ensure_tuple_rep([1, 2], 3)
        ValueError: Sequence must have length 3, got length 2.

    """
    if not issequenceiterable(tup):
        return (tup,) * dim
    elif len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")


def fall_back_tuple(user_provided: Any, default: Sequence, func: Callable = lambda x: x and x > 0) -> Tuple[Any, ...]:
    """
    Refine `user_provided` according to the `default`, and returns as a validated tuple.

    The validation is done for each element in `user_provided` using `func`.
    If `func(user_provided[idx])` returns False, the corresponding `default[idx]` will be used
    as the fallback.

    Typically used when `user_provided` is a tuple of window size provided by the user,
    `default` is defined by data, this function returns an updated `user_provided` with its non-positive
    components replaced by the corresponding components from `default`.

    Args:
        user_provided: item to be validated.
        default: a sequence used to provided the fallbacks.
        func: a Callable to validate every components of `user_provided`.

    Examples::

        >>> fall_back_tuple((1, 2), (32, 32))
        (1, 2)
        >>> fall_back_tuple(None, (32, 32))
        (32, 32)
        >>> fall_back_tuple((-1, 10), (32, 32))
        (32, 10)
        >>> fall_back_tuple((-1, None), (32, 32))
        (32, 32)
        >>> fall_back_tuple((1, None), (32, 32))
        (1, 32)
        >>> fall_back_tuple(0, (32, 32))
        (32, 32)
        >>> fall_back_tuple(range(3), (32, 64, 48))
        (32, 1, 2)
        >>> fall_back_tuple([0], (32, 32))
        ValueError: Sequence must have length 2, got length 1.

    """
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may not scale.

    See Also

        :py:class:`monai.transforms.Compose`
    """

    @abstractmethod
    def __call__(self, data: Any):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string
        - the data shape can be:

          #. string data without shape, `LoadNifti` and `LoadPNG` transforms expect file paths
          #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirst` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
          #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class Randomizable(ABC):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    This is mainly for randomized data augmentation transforms. For example::

        class RandShiftIntensity(Randomizable):
            def randomize():
                self._offset = self.R.uniform(low=0, high=100)
            def __call__(self, img):
                self.randomize()
                return img + self._offset

        transform = RandShiftIntensity()
        transform.set_random_state(seed=0)

    """

    R: np.random.RandomState = np.random.RandomState()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Raises:
            TypeError: When ``state`` is not an ``Optional[np.random.RandomState]``.

        Returns:
            a Randomizable instance.

        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % MAX_SEED
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    @abstractmethod
    def randomize(self, data: Any) -> None:
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


def rand_choice(prob: float = 0.5) -> bool:
    """
    Returns True if a randomly chosen number is less than or equal to `prob`, by default this is a 50/50 chance.
    """
    return bool(random.random() <= prob)


def img_bounds(img: np.ndarray) -> np.ndarray:
    """
    Returns the minimum and maximum indices of non-zero lines in axis 0 of `img`, followed by that for axis 1.
    """
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))


def in_bounds(x: float, y: float, margin: float, maxx: float, maxy: float) -> bool:
    """
    Returns True if (x,y) is within the rectangle (margin, margin, maxx-margin, maxy-margin).
    """
    return bool(margin <= x < (maxx - margin) and margin <= y < (maxy - margin))


def is_empty(img: Union[np.ndarray, torch.Tensor]) -> bool:
    """
    Returns True if `img` is empty, that is its maximum value is not greater than its minimum.
    """
    return not (img.max() > img.min())  # use > instead of <= so that an image full of NaNs will result in True


def zero_margins(img: np.ndarray, margin: int) -> bool:
    """
    Returns True if the values within `margin` indices of the edges of `img` in dimensions 1 and 2 are 0.
    """
    if np.any(img[:, :, :margin]) or np.any(img[:, :, -margin:]):
        return False

    if np.any(img[:, :margin, :]) or np.any(img[:, -margin:, :]):
        return False

    return True


def rescale_array(
    arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: Optional[np.dtype] = np.float32
) -> np.ndarray:
    """
    Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
    """
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def rescale_instance_array(
    arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescale each array slice along the first dimension of `arr` independently.
    """
    out: np.ndarray = np.zeros(arr.shape, dtype)
    for i in range(arr.shape[0]):
        out[i] = rescale_array(arr[i], minv, maxv, dtype)

    return out


def rescale_array_int_max(arr: np.ndarray, dtype: np.dtype = np.uint16) -> np.ndarray:
    """
    Rescale the array `arr` to be between the minimum and maximum values of the type `dtype`.
    """
    info: np.iinfo = np.iinfo(dtype)
    return rescale_array(arr, info.min, info.max).astype(dtype)


def copypaste_arrays(
    src_shape,
    dest_shape,
    srccenter: Sequence[int],
    destcenter: Sequence[int],
    dims: Sequence[Optional[int]],
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    """
    Calculate the slices to copy a sliced area of array in `src_shape` into array in `dest_shape`.

    The area has dimensions `dims` (use 0 or None to copy everything in that dimension),
    the source area is centered at `srccenter` index in `src` and copied into area centered at `destcenter` in `dest`.
    The dimensions of the copied area will be clipped to fit within the
    source and destination arrays so a smaller area may be copied than expected. Return value is the tuples of slice
    objects indexing the copied area in `src`, and those indexing the copy area in `dest`.

    Example

    .. code-block:: python

        src_shape = (6,6)
        src = np.random.randint(0,10,src_shape)
        dest = np.zeros_like(src)
        srcslices, destslices = copypaste_arrays(src_shape, dest.shape, (3, 2),(2, 1),(3, 4))
        dest[destslices] = src[srcslices]
        print(src)
        print(dest)

        >>> [[9 5 6 6 9 6]
             [4 3 5 6 1 2]
             [0 7 3 2 4 1]
             [3 0 0 1 5 1]
             [9 4 7 1 8 2]
             [6 6 5 8 6 7]]
            [[0 0 0 0 0 0]
             [7 3 2 4 0 0]
             [0 0 1 5 0 0]
             [4 7 1 8 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]

    """
    s_ndim = len(src_shape)
    d_ndim = len(dest_shape)
    srcslices = [slice(None)] * s_ndim
    destslices = [slice(None)] * d_ndim

    for i, ss, ds, sc, dc, dim in zip(range(s_ndim), src_shape, dest_shape, srccenter, destcenter, dims):
        if dim:
            # dimension before midpoint, clip to size fitting in both arrays
            d1 = np.clip(dim // 2, 0, min(sc, dc))
            # dimension after midpoint, clip to size fitting in both arrays
            d2 = np.clip(dim // 2 + 1, 0, min(ss - sc, ds - dc))

            srcslices[i] = slice(sc - d1, sc + d2)
            destslices[i] = slice(dc - d1, dc + d2)

    return tuple(srcslices), tuple(destslices)


def resize_center(
    img: np.ndarray, *resize_dims: Optional[int], fill_value: float = 0.0, inplace: bool = True
) -> np.ndarray:
    """
    Resize `img` by cropping or expanding the image from the center. The `resize_dims` values are the output dimensions
    (or None to use original dimension of `img`). If a dimension is smaller than that of `img` then the result will be
    cropped and if larger padded with zeros, in both cases this is done relative to the center of `img`. The result is
    a new image with the specified dimensions and values from `img` copied into its center.
    """

    resize_dims = fall_back_tuple(resize_dims, img.shape)

    half_img_shape = np.asarray(img.shape) // 2
    half_dest_shape = np.asarray(resize_dims) // 2
    srcslices, destslices = copypaste_arrays(img.shape, resize_dims, half_img_shape, half_dest_shape, resize_dims)

    if not inplace:
        dest = np.full(resize_dims, fill_value, img.dtype)
        dest[destslices] = img[srcslices]
        return dest
    return img[srcslices]


def map_binary_to_indices(
    label: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [1, 0, 1], [1, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 3, 5, 6, 7])`` and ``background indices = np.array([0, 4, 8])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.

    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
    fg_indices = np.nonzero(label_flat)[0]
    if image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
    else:
        bg_indices = np.nonzero(~label_flat)[0]
    return fg_indices, bg_indices


def weighted_patch_samples(
    spatial_size: Union[int, Sequence[int]],
    w: np.ndarray,
    n_samples: int = 1,
    r_state: Optional[np.random.RandomState] = None,
) -> List:
    """
    Computes `n_samples` of random patch sampling locations, given the sampling weight map `w` and patch `spatial_size`.

    Args:
        spatial_size: length of each spatial dimension of the patch.
        w: weight map, the weights must be non-negative. each element denotes a sampling weight of the spatial location.
            0 indicates no sampling.
            The weight map shape is assumed ``(spatial_dim_0, spatial_dim_1, ..., spatial_dim_n)``.
        n_samples: number of patch samples
        r_state: a random state container

    Returns:
        a list of `n_samples` N-D integers representing the spatial sampling location of patches.

    """
    if w is None:
        raise ValueError("w must be an ND array.")
    if r_state is None:
        r_state = np.random.RandomState()
    img_size = np.asarray(w.shape, dtype=np.int)
    win_size = np.asarray(fall_back_tuple(spatial_size, img_size), dtype=np.int)

    s = tuple(slice(w // 2, m - w + w // 2) if m > w else slice(m // 2, m // 2 + 1) for w, m in zip(win_size, img_size))
    v = w[s]  # weight map in the 'valid' mode
    v_size = v.shape
    v = v.ravel()
    if np.any(v < 0):
        v -= np.min(v)  # shifting to non-negative
    v = v.cumsum()
    if not v[-1] or not np.isfinite(v[-1]) or v[-1] < 0:  # uniform sampling
        idx = r_state.randint(0, len(v), size=n_samples)
    else:
        idx = v.searchsorted(r_state.random(n_samples) * v[-1], side="right")
    # compensate 'valid' mode
    diff = np.minimum(win_size, img_size) // 2
    centers = [np.unravel_index(i, v_size) + diff for i in np.asarray(idx, dtype=np.int)]
    return centers


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: np.ndarray,
    bg_indices: np.ndarray,
    rand_state: np.random.RandomState = np.random,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The proposed roi is larger than the image.")

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
        if valid_start[i] == valid_end[i]:
            valid_end[i] += 1

    # noinspection PyShadowingNames
    def _correct_centers(
        center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
    ) -> List[np.ndarray]:
        for i, c in enumerate(center_ori):
            center_i = c
            if c < valid_start[i]:
                center_i = valid_start[i]
            if c >= valid_end[i]:
                center_i = valid_end[i] - 1
            center_ori[i] = center_i
        return center_ori

    centers = []

    if not len(fg_indices) or not len(bg_indices):
        if not len(fg_indices) and not len(bg_indices):
            raise ValueError("No sampling location available.")
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if not len(fg_indices) else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(_correct_centers(center_ori, valid_start, valid_end))

    return centers


def apply_transform(transform: Callable, data, map_items: bool = True):
    """
    Transform `data` with `transform`.
    If `data` is a list or tuple and `map_data` is True, each item of `data` will be transformed
    and this method returns a list of outcomes.
    otherwise transform will be applied once with `data` as the argument.

    Args:
        transform: a callable to be used to transform `data`
        data: an object to be transformed.
        map_items: whether to apply transform to each item in `data`,
            if `data` is a list or tuple. Defaults to True.

    Raises:
        Exception: When ``transform`` raises an exception.

    """
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise RuntimeError(f"applying transform {transform}") from e


def create_grid(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype: np.dtype = float,
) -> np.ndarray:
    """
    compute a `spatial_size` mesh.

    Args:
        spatial_size: spatial size of the grid.
        spacing: same len as ``spatial_size``, defaults to 1.0 (dense grid).
        homogeneous: whether to make homogeneous coordinates.
        dtype: output grid data type.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [np.linspace(-(d - 1.0) / 2.0 * s, (d - 1.0) / 2.0 * s, int(d)) for d, s in zip(spatial_size, spacing)]
    coords = np.asarray(np.meshgrid(*ranges, indexing="ij"), dtype=dtype)
    if not homogeneous:
        return coords
    return np.concatenate([coords, np.ones_like(coords[:1])])


def create_control_grid(
    spatial_shape: Sequence[int], spacing: Sequence[float], homogeneous: bool = True, dtype: np.dtype = float
) -> np.ndarray:
    """
    control grid with two additional point in each direction
    """
    grid_shape = []
    for d, s in zip(spatial_shape, spacing):
        d = int(d)
        if d % 2 == 0:
            grid_shape.append(np.ceil((d - 1.0) / (2.0 * s) + 0.5) * 2.0 + 2.0)
        else:
            grid_shape.append(np.ceil((d - 1.0) / (2.0 * s)) * 2.0 + 3.0)
    return create_grid(grid_shape, spacing, homogeneous, dtype)


def create_rotate(spatial_dims: int, radians: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a 2D or 3D rotation matrix

    Args:
        spatial_dims: {``2``, ``3``} spatial rank
        radians: rotation radians
            when spatial_dims == 3, the `radians` sequence corresponds to
            rotation in the 1st, 2nd, and 3rd dim respectively.

    Raises:
        ValueError: When ``radians`` is empty.
        ValueError: When ``spatial_dims`` is not one of [2, 3].

    """
    radians = ensure_tuple(radians)
    if spatial_dims == 2:
        if len(radians) >= 1:
            sin_, cos_ = np.sin(radians[0]), np.cos(radians[0])
            return np.array([[cos_, -sin_, 0.0], [sin_, cos_, 0.0], [0.0, 0.0, 1.0]])
        raise ValueError("radians must be non empty.")

    if spatial_dims == 3:
        affine = None
        if len(radians) >= 1:
            sin_, cos_ = np.sin(radians[0]), np.cos(radians[0])
            affine = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, cos_, -sin_, 0.0], [0.0, sin_, cos_, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if len(radians) >= 2:
            sin_, cos_ = np.sin(radians[1]), np.cos(radians[1])
            affine = affine @ np.array(
                [[cos_, 0.0, sin_, 0.0], [0.0, 1.0, 0.0, 0.0], [-sin_, 0.0, cos_, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if len(radians) >= 3:
            sin_, cos_ = np.sin(radians[2]), np.cos(radians[2])
            affine = affine @ np.array(
                [[cos_, -sin_, 0.0, 0.0], [sin_, cos_, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        if affine is None:
            raise ValueError("radians must be non empty.")
        return affine

    raise ValueError(f"Unsupported spatial_dims: {spatial_dims}, available options are [2, 3].")


def create_shear(spatial_dims: int, coefs: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a shearing matrix

    Args:
        spatial_dims: spatial rank
        coefs: shearing factors, defaults to 0.

    Raises:
        NotImplementedError: When ``spatial_dims`` is not one of [2, 3].

    """
    if spatial_dims == 2:
        coefs = ensure_tuple_size(coefs, dim=2, pad_val=0.0)
        return np.array([[1, coefs[0], 0.0], [coefs[1], 1.0, 0.0], [0.0, 0.0, 1.0]])
    if spatial_dims == 3:
        coefs = ensure_tuple_size(coefs, dim=6, pad_val=0.0)
        return np.array(
            [
                [1.0, coefs[0], coefs[1], 0.0],
                [coefs[2], 1.0, coefs[3], 0.0],
                [coefs[4], coefs[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    raise NotImplementedError("Currently only spatial_dims in [2, 3] are supported.")


def create_scale(spatial_dims: int, scaling_factor: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a scaling matrix

    Args:
        spatial_dims: spatial rank
        scaling_factor: scaling factors, defaults to 1.
    """
    scaling_factor = ensure_tuple_size(scaling_factor, dim=spatial_dims, pad_val=1.0)
    return np.diag(scaling_factor[:spatial_dims] + (1.0,))


def create_translate(spatial_dims: int, shift: Union[Sequence[float], float]) -> np.ndarray:
    """
    create a translation matrix

    Args:
        spatial_dims: spatial rank
        shift: translate factors, defaults to 0.
    """
    shift = ensure_tuple(shift)
    affine = np.eye(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return affine


def generate_spatial_bounding_box(
    img: np.ndarray,
    select_fn: Callable = lambda x: x > 0,
    channel_indices: Optional[IndexSelection] = None,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img[[*(ensure_tuple(channel_indices))]] if channel_indices is not None else img
    data = np.any(select_fn(data), axis=0)
    nonzero_idx = np.nonzero(data)
    margin = ensure_tuple_rep(margin, data.ndim)

    box_start = list()
    box_end = list()
    for i in range(data.ndim):
        assert len(nonzero_idx[i]) > 0, f"did not find nonzero index at spatial dim {i}"
        box_start.append(max(0, np.min(nonzero_idx[i]) - margin[i]))
        box_end.append(min(data.shape[i], np.max(nonzero_idx[i]) + margin[i] + 1))
    return box_start, box_end


class InterpolateMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    AREA = "area"


class GridSampleMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    FOURTH = "fourth"


class GridSamplePadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    ZEROS = "zeros"
    BORDER = "border"
    REFLECTION = "reflection"


class BlendMode(Enum):
    """
    See also: :py:class:`monai.data.utils.compute_importance_map`
    """

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PytorchPadMode(Enum):
    """
    See also: https://pytorch.org/docs/stable/nn.functional.html#pad
    """

    CONSTANT = "constant"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"


def normalize_transform(
    shape: Sequence[int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Compute an affine matrix according to the input shape.
    The transform normalizes the homogeneous image coordinates to the
    range of `[-1, 1]`.

    Args:
        shape: input spatial shape
        device: device on which the returned affine will be allocated.
        dtype: data type of the returned affine
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
    """
    norm = torch.tensor(shape, dtype=torch.float64, device=device)  # no in-place change
    if align_corners:
        norm[norm <= 1.0] = 2.0
        norm = 2.0 / (norm - 1.0)
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = -1.0
    else:
        norm[norm <= 0.0] = 2.0
        norm = 2.0 / norm
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = 1.0 / torch.tensor(shape, dtype=torch.float64, device=device) - 1.0
    norm = norm.unsqueeze(0).to(dtype=dtype)
    norm.requires_grad = False
    return norm


def to_norm_affine(
    affine: torch.Tensor, src_size: Sequence[int], dst_size: Sequence[int], align_corners: bool = False
) -> torch.Tensor:
    """
    Given ``affine`` defined for coordinates in the pixel space, compute the corresponding affine
    for the normalized coordinates.

    Args:
        affine: Nxdxd batched square matrix
        src_size: source image spatial shape
        dst_size: target image spatial shape
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample

    Raises:
        TypeError: When ``affine`` is not a ``torch.Tensor``.
        ValueError: When ``affine`` is not Nxdxd.
        ValueError: When ``src_size`` or ``dst_size`` dimensions differ from ``affine``.

    """
    if not torch.is_tensor(affine):
        raise TypeError(f"affine must be a torch.Tensor but is {type(affine).__name__}.")
    if affine.ndimension() != 3 or affine.shape[1] != affine.shape[2]:
        raise ValueError(f"affine must be Nxdxd, got {tuple(affine.shape)}.")
    sr = affine.shape[1] - 1
    if sr != len(src_size) or sr != len(dst_size):
        raise ValueError(f"affine suggests {sr}D, got src={len(src_size)}D, dst={len(dst_size)}D.")

    src_xform = normalize_transform(src_size, affine.device, affine.dtype, align_corners)
    dst_xform = normalize_transform(dst_size, affine.device, affine.dtype, align_corners)
    new_affine = src_xform @ affine @ torch.inverse(dst_xform)
    return new_affine


# noinspection PyTypeChecker
class AffineTransform(nn.Module):
    def __init__(
        self,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        normalized: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.ZEROS,
        align_corners: bool = False,
        reverse_indexing: bool = False,
    ) -> None:
        """
        Apply affine transformations with a batch of affine matrices.

        When `normalized=False` and `reverse_indexing=True`,
        it does the commonly used resampling in the 'pull' direction
        following the ``scipy.ndimage.affine_transform`` convention.
        In this case `theta` is equivalent to (ndim+1, ndim+1) input ``matrix`` of ``scipy.ndimage.affine_transform``,
        operates on homogeneous coordinates.
        See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html

        When `normalized=True` and `reverse_indexing=False`,
        it applies `theta` to the normalized coordinates (coords. in the range of [-1, 1]) directly.
        This is often used with `align_corners=False` to achieve resolution-agnostic resampling,
        thus useful as a part of trainable modules such as the spatial transformer networks.
        See also: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

        Args:
            spatial_size: output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src` input of `self.forward`.
            normalized: indicating whether the provided affine matrix `theta` is defined
                for the normalized coordinates. If `normalized=False`, `theta` will be converted
                to operate on normalized coordinates as pytorch affine_grid works with the normalized
                coordinates.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: see also https://pytorch.org/docs/stable/nn.functional.html#grid-sample.
            reverse_indexing: whether to reverse the spatial indexing of image and coordinates.
                set to `False` if `theta` follows pytorch's default "D, H, W" convention.
                set to `True` if `theta` follows `scipy.ndimage` default "i, j, k" convention.
        """
        super().__init__()
        self.spatial_size = ensure_tuple(spatial_size) if spatial_size is not None else None
        self.normalized = normalized
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.reverse_indexing = reverse_indexing

    def forward(
        self, src: torch.Tensor, theta: torch.Tensor, spatial_size: Optional[Union[Sequence[int], int]] = None
    ) -> torch.Tensor:
        """
        ``theta`` must be an affine transformation matrix with shape
        3x3 or Nx3x3 or Nx2x3 or 2x3 for spatial 2D transforms,
        4x4 or Nx4x4 or Nx3x4 or 3x4 for spatial 3D transforms,
        where `N` is the batch size. `theta` will be converted into float Tensor for the computation.

        Args:
            src (array_like): image in spatial 2D or 3D (N, C, spatial_dims),
                where N is the batch dim, C is the number of channels.
            theta (array_like): Nx3x3, Nx2x3, 3x3, 2x3 for spatial 2D inputs,
                Nx4x4, Nx3x4, 3x4, 4x4 for spatial 3D inputs. When the batch dimension is omitted,
                `theta` will be repeated N times, N is the batch dim of `src`.
            spatial_size: output spatial shape, the full output shape will be
                `[N, C, *spatial_size]` where N and C are inferred from the `src`.

        Raises:
            TypeError: When ``theta`` is not a ``torch.Tensor``.
            ValueError: When ``theta`` is not one of [Nxdxd, dxd].
            ValueError: When ``theta`` is not one of [Nx3x3, Nx4x4].
            TypeError: When ``src`` is not a ``torch.Tensor``.
            ValueError: When ``src`` spatially is not one of [2D, 3D].
            ValueError: When affine and image batch dimension differ.

        """
        # validate `theta`
        if not torch.is_tensor(theta):
            raise TypeError(f"theta must be torch.Tensor but is {type(theta).__name__}.")
        if theta.dim() not in (2, 3):
            raise ValueError(f"theta must be Nxdxd or dxd, got {theta.shape}.")
        if theta.dim() == 2:
            theta = theta[None]  # adds a batch dim.
        theta = theta.clone()  # no in-place change of theta
        theta_shape = tuple(theta.shape[1:])
        if theta_shape in ((2, 3), (3, 4)):  # needs padding to dxd
            pad_affine = torch.tensor([0, 0, 1] if theta_shape[0] == 2 else [0, 0, 0, 1])
            pad_affine = pad_affine.repeat(theta.shape[0], 1, 1).to(theta)
            pad_affine.requires_grad = False
            theta = torch.cat([theta, pad_affine], dim=1)
        if tuple(theta.shape[1:]) not in ((3, 3), (4, 4)):
            raise ValueError(f"theta must be Nx3x3 or Nx4x4, got {theta.shape}.")

        # validate `src`
        if not torch.is_tensor(src):
            raise TypeError(f"src must be torch.Tensor but is {type(src).__name__}.")
        sr = src.dim() - 2  # input spatial rank
        if sr not in (2, 3):
            raise ValueError(f"Unsupported src dimension: {sr}, available options are [2, 3].")

        # set output shape
        src_size = tuple(src.shape)
        dst_size = src_size  # default to the src shape
        if self.spatial_size is not None:
            dst_size = src_size[:2] + self.spatial_size
        if spatial_size is not None:
            dst_size = src_size[:2] + ensure_tuple(spatial_size)

        # reverse and normalise theta if needed
        if not self.normalized:
            theta = to_norm_affine(
                affine=theta, src_size=src_size[2:], dst_size=dst_size[2:], align_corners=self.align_corners
            )
        if self.reverse_indexing:
            rev_idx = torch.as_tensor(range(sr - 1, -1, -1), device=src.device)
            theta[:, :sr] = theta[:, rev_idx]
            theta[:, :, :sr] = theta[:, :, rev_idx]
        if (theta.shape[0] == 1) and src_size[0] > 1:
            # adds a batch dim to `theta` in order to match `src`
            theta = theta.repeat(src_size[0], 1, 1)
        if theta.shape[0] != src_size[0]:
            raise ValueError(
                f"affine and image batch dimension must match, got affine={theta.shape[0]} image={src_size[0]}."
            )

        grid = nn.functional.affine_grid(theta=theta[:, :sr], size=list(dst_size), align_corners=self.align_corners)
        dst = nn.functional.grid_sample(
            input=src.contiguous(),
            grid=grid,
            mode=self.mode.value,
            padding_mode=self.padding_mode.value,
            align_corners=self.align_corners,
        )
        return dst


class AffineGrid(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: angle range in radians. rotate_params[0] with be used to generate the 1st rotation
            parameter from `uniform[-rotate_params[0], rotate_params[0])`. Similarly, `rotate_params[1]` and
            `rotate_params[2]` are used in 3D affine for the range of 2nd and 3rd axes.
        shear_params: shear_params[0] with be used to generate the 1st shearing parameter from
            `uniform[-shear_params[0], shear_params[0])`. Similarly, `shear_params[1]` to
            `shear_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        translate_params : translate_params[0] with be used to generate the 1st shift parameter from
            `uniform[-translate_params[0], translate_params[0])`. Similarly, `translate_params[1]`
            to `translate_params[N]` controls the range of the uniform distribution used to generate
            the 2nd to N-th parameter.
        scale_params: scale_params[0] with be used to generate the 1st scaling factor from
            `uniform[-scale_params[0], scale_params[0]) + 1.0`. Similarly, `scale_params[1]` to
            `scale_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.

    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params

        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")

        spatial_dims = len(grid.shape) - 1
        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            affine = affine @ create_rotate(spatial_dims, self.rotate_params)
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            affine = affine @ create_translate(spatial_dims, self.translate_params)
        if self.scale_params:
            affine = affine @ create_scale(spatial_dims, self.scale_params)
        affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)

        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
        grid = (affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if self.as_tensor_output:
            return grid
        return grid.cpu().numpy()


class Resample(Transform):
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        if not torch.is_tensor(img):
            img = torch.as_tensor(np.ascontiguousarray(img))
        assert grid is not None, "Error, grid argument must be supplied as an ndarray or tensor "
        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        if USE_COMPILED:
            raise NotImplementedError('This is a modified version; check MONAI reference page for complied version.')
        else:
            for i, dim in enumerate(img.shape[1:]):
                grid[i] = 2.0 * grid[i] / (dim - 1.0)
            grid = grid[:-1] / grid[-1:]
            index_ordering: List[int] = list(range(img.ndimension() - 2, -1, -1))
            grid = grid[index_ordering]
            grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
            out = torch.nn.functional.grid_sample(
                img.unsqueeze(0).float(),
                grid.unsqueeze(0).float(),
                mode=self.mode.value if mode is None else GridSampleMode(mode).value,
                padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
                align_corners=True,
            )[0]
        if self.as_tensor_output:
            return out
        return out.cpu().numpy()


class Affine(Transform):
    """
    Transform ``img`` given the affine parameters.
    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Defaults to no scaling.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
        """
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        grid = self.affine_grid(spatial_size=sp_size)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )


class Rotate(Transform):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
    """

    def __init__(
        self,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Optional[np.dtype] = np.float64,
    ) -> None:
        self.angle = angle
        self.keep_size = keep_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: np.ndarray,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        _dtype = dtype or self.dtype or img.dtype
        im_shape = np.asarray(img.shape[1:])  # spatial dimensions
        input_ndim = len(im_shape)
        if input_ndim not in (2, 3):
            raise ValueError(f"Unsupported img dimension: {input_ndim}, available options are [2, 3].")
        _angle = ensure_tuple_rep(self.angle, 1 if input_ndim == 2 else 3)
        transform = create_rotate(input_ndim, _angle)
        shift = create_translate(input_ndim, (im_shape - 1) / 2)
        if self.keep_size:
            output_shape = im_shape
        else:
            corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape(
                (len(im_shape), -1)
            )
            corners = transform[:-1, :-1] @ corners
            output_shape = (corners.ptp(axis=1) + 0.5).astype(int)
        shift_1 = create_translate(input_ndim, -(output_shape - 1) / 2)
        transform = shift @ transform @ shift_1

        xform = AffineTransform(
            normalized=False,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
            align_corners=self.align_corners if align_corners is None else align_corners,
            reverse_indexing=True,
        )
        output = xform(
            torch.as_tensor(np.ascontiguousarray(img).astype(_dtype)).unsqueeze(0),
            torch.as_tensor(np.ascontiguousarray(transform).astype(_dtype)),
            spatial_size=output_shape,
        )
        output = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return output


class NormalizeIntensity(Transform):
    """
    Normalize input based on provided args, using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        subtrahend: Optional[Sequence] = None,
        divisor: Optional[Sequence] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img: np.ndarray, sub=None, div=None) -> np.ndarray:
        slices = (img != 0) if self.nonzero else np.ones(img.shape, dtype=np.bool_)
        if not np.any(slices):
            return img

        _sub = sub if sub is not None else np.mean(img[slices])
        if isinstance(_sub, np.ndarray):
            _sub = _sub[slices]

        _div = div if div is not None else np.std(img[slices])
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, np.ndarray):
            _div = _div[slices]
            _div[_div == 0.0] = 1.0
        img[slices] = (img[slices] - _sub) / _div
        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components.")
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.divisor)} components.")

            for i, d in enumerate(img):
                img[i] = self._normalize(
                    d,
                    sub=self.subtrahend[i] if self.subtrahend is not None else None,
                    div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            img = self._normalize(img, self.subtrahend, self.divisor)

        return img


class ScaleIntensityRange(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
    """

    def __init__(self, a_min: float, a_max: float, b_min: float, b_max: float, clip: bool = False) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        if self.a_max - self.a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            return img - self.a_min + self.b_min

        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img


class ScaleIntensityRangePercentiles(Transform):
    """
    Apply range scaling to a numpy array based on the intensity distribution of the input.

    By default this transform will scale from [lower_intensity_percentile, upper_intensity_percentile] to [b_min, b_max], where
    {lower,upper}_intensity_percentile are the intensity values at the corresponding percentiles of ``img``.

    The ``relative`` parameter can also be set to scale from [lower_intensity_percentile, upper_intensity_percentile] to the
    lower and upper percentiles of the output range [b_min, b_max]

    For example:

    .. code-block:: python
        :emphasize-lines: 11, 22

        image = np.array(
            [[[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]]])

        # Scale from lower and upper image intensity percentiles
        # to output range [b_min, b_max]
        scaler = ScaleIntensityRangePercentiles(10, 90, 0, 200, False, False)
        print(scaler(image))
        [[[0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.]]]

        # Scale from lower and upper image intensity percentiles
        # to lower and upper percentiles of the output range [b_min, b_max]
        rel_scaler = ScaleIntensityRangePercentiles(10, 90, 0, 200, False, True)
        print(rel_scaler(image))
        [[[20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.]]]


    Args:
        lower: lower intensity percentile.
        upper: upper intensity percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max].
    """

    def __init__(
        self, lower: float, upper: float, b_min: float, b_max: float, clip: bool = False, relative: bool = False
    ) -> None:
        assert 0.0 <= lower <= 100.0, "Percentiles must be in the range [0, 100]"
        assert 0.0 <= upper <= 100.0, "Percentiles must be in the range [0, 100]"
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.relative = relative

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        a_min = np.percentile(img, self.lower)
        a_max = np.percentile(img, self.upper)
        b_min = self.b_min
        b_max = self.b_max

        if self.relative:
            b_min = ((self.b_max - self.b_min) * (self.lower / 100.0)) + self.b_min
            b_max = ((self.b_max - self.b_min) * (self.upper / 100.0)) + self.b_min

        scalar = ScaleIntensityRange(a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=False)
        img = scalar(img)

        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img


class MaskIntensity(Transform):
    """
    Mask the intensity values of input image with the specified mask data.
    Mask data must have the same spatial size as the input image, and all
    the intensity values of input image corresponding to `0` in the mask
    data will be set to `0`, others will keep the original value.

    Args:
        mask_data: if mask data is single channel, apply to evey channel
            of input image. if multiple channels, the channel number must
            match input data. mask_data will be converted to `bool` values
            by `mask_data > 0` before applying transform to input image.

    """

    def __init__(self, mask_data: np.ndarray) -> None:
        self.mask_data = mask_data

    def __call__(self, img: np.ndarray, mask_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            mask_data: if mask data is single channel, apply to evey channel
                of input image. if multiple channels, the channel number must
                match input data. mask_data will be converted to `bool` values
                by `mask_data > 0` before applying transform to input image.

        Raises:
            ValueError: When ``mask_data`` and ``img`` channels differ and ``mask_data`` is not single channel.

        """
        mask_data_ = self.mask_data > 0 if mask_data is None else mask_data > 0
        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img={img.shape[0]} mask_data={mask_data_.shape[0]}."
            )

        return img * mask_data_


# ---------------------------------------------------------------
# ------------------------- CROP AND PAD ------------------------
# ---------------------------------------------------------------


class SpatialPad(Transform):
    """
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
    for additional details.

    Args:
        spatial_size: the spatial size of output data after padding.
            If its components have non-positive values, the corresponding size of input image will be used (no padding).
        method: {``"symmetric"``, ``"end"``}
            Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
    ) -> None:
        self.spatial_size = spatial_size
        self.method: Method = Method(method)
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def _determine_data_pad_width(self, data_shape: Sequence[int]) -> List[Tuple[int, int]]:
        self.spatial_size = fall_back_tuple(self.spatial_size, data_shape)
        if self.method == Method.SYMMETRIC:
            pad_width = list()
            for i in range(len(self.spatial_size)):
                width = max(self.spatial_size[i] - data_shape[i], 0)
                pad_width.append((width // 2, width - (width // 2)))
            return pad_width
        else:
            return [(0, max(self.spatial_size[i] - data_shape[i], 0)) for i in range(len(self.spatial_size))]

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        data_pad_width = self._determine_data_pad_width(img.shape[1:])
        all_pad_width = [(0, 0)] + data_pad_width
        if not np.asarray(all_pad_width).any():
            # all zeros, skip padding
            return img
        else:
            img = np.pad(img, all_pad_width, mode=self.mode.value if mode is None else NumpyPadMode(mode).value)
            return img


class BorderPad(Transform):
    """
    Pad the input data by adding specified borders to every dimension.

    Args:
        spatial_border: specified size for every spatial border. it can be 3 shapes:

            - single int number, pad all the borders with the same size.
            - length equals the length of image shape, pad every spatial dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [2, 1],
              pad every border of H dim with 2, pad every border of W dim with 1, result shape is [1, 8, 6].
            - length equals 2 x (length of image shape), pad every border of every dimension separately.
              for example, image shape(CHW) is [1, 4, 4], spatial_border is [1, 2, 3, 4], pad top of H dim with 1,
              pad bottom of H dim with 2, pad left of W dim with 3, pad right of W dim with 4.
              the result shape is [1, 7, 11].

        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    def __init__(
        self, spatial_border: Union[Sequence[int], int], mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT
    ) -> None:
        self.spatial_border = spatial_border
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first and
                padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        Raises:
            ValueError: When ``self.spatial_border`` contains a nonnegative int.
            ValueError: When ``self.spatial_border`` length is not one of
                [1, len(spatial_shape), 2*len(spatial_shape)].

        """
        spatial_shape = img.shape[1:]
        spatial_border = ensure_tuple(self.spatial_border)
        for b in spatial_border:
            if not isinstance(b, int) or b < 0:
                raise ValueError(f"self.spatial_border must contain only nonnegative ints, got {spatial_border}.")

        if len(spatial_border) == 1:
            data_pad_width = [(spatial_border[0], spatial_border[0]) for _ in range(len(spatial_shape))]
        elif len(spatial_border) == len(spatial_shape):
            data_pad_width = [(spatial_border[i], spatial_border[i]) for i in range(len(spatial_shape))]
        elif len(spatial_border) == len(spatial_shape) * 2:
            data_pad_width = [(spatial_border[2 * i], spatial_border[2 * i + 1]) for i in range(len(spatial_shape))]
        else:
            raise ValueError(
                f"Unsupported spatial_border length: {len(spatial_border)}, available options are "
                f"[1, len(spatial_shape)={len(spatial_shape)}, 2*len(spatial_shape)={2*len(spatial_shape)}]."
            )

        return np.pad(
            img, [(0, 0)] + data_pad_width, mode=self.mode.value if mode is None else NumpyPadMode(mode).value
        )


class DivisiblePad(Transform):
    """
    Pad the input data, so that the spatial sizes are divisible by `k`.
    """

    def __init__(self, k: Union[Sequence[int], int], mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT) -> None:
        """
        Args:
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        See also :py:class:`monai.transforms.SpatialPad`
        """
        self.k = k
        self.mode: NumpyPadMode = NumpyPadMode(mode)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to be transformed, assuming `img` is channel-first
                and padding doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``self.mode``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        spatial_shape = img.shape[1:]
        k = fall_back_tuple(self.k, (1,) * len(spatial_shape))
        new_size = []
        for k_d, dim in zip(k, spatial_shape):
            new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
            new_size.append(new_dim)

        spatial_pad = SpatialPad(spatial_size=new_size, method=Method.SYMMETRIC, mode=mode or self.mode)
        return spatial_pad(img)


class SpatialCrop(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            self.roi_end = np.maximum(self.roi_start + roi_size, self.roi_start)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.roi_start), len(self.roi_end), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
        return img[tuple(slices)]


class CenterSpatialCrop(Transform):
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, roi_size: Union[Sequence[int], int]) -> None:
        self.roi_size = roi_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        center = [i // 2 for i in img.shape[1:]]
        cropper = SpatialCrop(roi_center=center, roi_size=self.roi_size)
        return cropper(img)


class RandSpatialCrop(Randomizable, Transform):
    """
    Crop image with random size or specific size ROI. It can crop at a random position as center
    or at the image center. And allows to set the minimum size to limit the randomly generated ROI.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            If its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """

    def __init__(
        self, roi_size: Union[Sequence[int], int], random_center: bool = True, random_size: bool = True
    ) -> None:
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size
        self._size: Optional[Sequence[int]] = None
        self._slices: Optional[Tuple[slice, ...]] = None

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            self._size = tuple((self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))))
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.randomize(img.shape[1:])
        assert self._size is not None
        if self.random_center:
            return img[self._slices]
        else:
            cropper = CenterSpatialCrop(self._size)
            return cropper(img)


class RandSpatialCropSamples(Randomizable, Transform):
    """
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI.
    It will return a list of cropped images.

    Args:
        roi_size: if `random_size` is True, the spatial size of the minimum crop region.
            if `random_size` is False, specify the expected ROI size to crop. e.g. [224, 224, 128]
        num_samples: number of samples (crop regions) to take in the returned list.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        num_samples: int,
        random_center: bool = True,
        random_size: bool = True,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.num_samples = num_samples
        self.cropper = RandSpatialCrop(roi_size, random_center, random_size)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        super().set_random_state(seed=seed, state=state)
        self.cropper.set_random_state(state=self.R)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        pass

    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        cropping doesn't change the channel dim.
        """
        return [self.cropper(img) for _ in range(self.num_samples)]


class CropForeground(Transform):
    """
    Crop an image using a bounding box. The bounding box is generated by selecting foreground using select_fn
    at channels channel_indices. margin is added in each spatial dimension of the bounding box.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box of foreground object.
    For example:

    .. code-block:: python

        image = np.array(
            [[[0, 0, 0, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 1, 3, 2, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 0, 0, 0]]])  # 1x5x5, single channel 5x5 image
        cropper = CropForeground(select_fn=lambda x: x > 1, margin=0)
        print(cropper(image))
        [[[2, 1],
          [3, 2],
          [2, 1]]]

    """

    def __init__(
        self,
        select_fn: Callable = lambda x: x > 0,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        return_coords: bool = False,
    ) -> None:
        """
        Args:
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            return_coords: whether return the coordinates of spatial bounding box for foreground.
        """
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.return_coords = return_coords

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = generate_spatial_bounding_box(img, self.select_fn, self.channel_indices, self.margin)
        cropped = SpatialCrop(roi_start=box_start, roi_end=box_end)(img)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped


class RandWeightedCrop(Randomizable, Transform):
    """
    Samples a list of `num_samples` image patches according to the provided `weight_map`.

    Args:
        spatial_size: the spatial size of the image patch e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `img` will be used.
        num_samples: number of samples (image patches) to take in the returned list.
        weight_map: weight map used to generate patch samples. The weights must be non-negative.
            Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
            It should be a single-channel array in shape, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`.
    """

    def __init__(
        self, spatial_size: Union[Sequence[int], int], num_samples: int = 1, weight_map: Optional[np.ndarray] = None
    ):
        self.spatial_size = ensure_tuple(spatial_size)
        self.num_samples = int(num_samples)
        self.weight_map = weight_map
        self.centers: List[np.ndarray] = []

    def randomize(self, weight_map: np.ndarray) -> None:
        self.centers = weighted_patch_samples(
            spatial_size=self.spatial_size, w=weight_map[0], n_samples=self.num_samples, r_state=self.R
        )  # using only the first channel as weight map

    def __call__(self, img: np.ndarray, weight_map: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Args:
            img: input image to sample patches from. assuming `img` is a channel-first array.
            weight_map: weight map used to generate patch samples. The weights must be non-negative.
                Each element denotes a sampling weight of the spatial location. 0 indicates no sampling.
                It should be a single-channel array in shape, for example, `(1, spatial_dim_0, spatial_dim_1, ...)`

        Returns:
            A list of image patches
        """
        if weight_map is None:
            weight_map = self.weight_map
        if weight_map is None:
            raise ValueError("weight map must be provided for weighted patch sampling.")
        if img.shape[1:] != weight_map.shape[1:]:
            raise ValueError(f"image and weight map spatial shape mismatch: {img.shape[1:]} vs {weight_map.shape[1:]}.")
        self.randomize(weight_map)
        _spatial_size = fall_back_tuple(self.spatial_size, weight_map.shape[1:])
        results = []
        for center in self.centers:
            cropper = SpatialCrop(roi_center=center, roi_size=_spatial_size)
            results.append(cropper(img))
        return results


class RandCropByPosNegLabel(Randomizable, Transform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if image is None:
            image = self.image
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices = self.fg_indices
                bg_indices = self.bg_indices
            else:
                fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        self.randomize(label, fg_indices, bg_indices, image)
        results: List[np.ndarray] = list()
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results


class ResizeWithPadOrCrop(Transform):
    """
    Resize an image to a target spatial size by either centrally cropping the image or
    padding it evenly with a user-specified mode.
    When the dimension is smaller than the target size, do symmetric padding along that dim.
    When the dimension is larger than the target size, do central cropping along that dim.

    Args:
        spatial_size: the spatial size of output data after padding or crop.
            If has non-positive values, the corresponding size of input image will be used (no padding).
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function for padding. Defaults to ``"constant"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        mode: Union[NumpyPadMode, str] = NumpyPadMode.CONSTANT,
    ):
        self.padder = SpatialPad(spatial_size=spatial_size, mode=mode)
        self.cropper = CenterSpatialCrop(roi_size=spatial_size)

    def __call__(self, img: np.ndarray, mode: Optional[Union[NumpyPadMode, str]] = None) -> np.ndarray:
        """
        Args:
            img: data to pad or crop, assuming `img` is channel-first and
                padding or cropping doesn't apply to the channel dim.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function for padding.
                If None, defaults to the ``mode`` in construction.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        """
        return self.padder(self.cropper(img), mode=mode)


# ---------------------------------------------------------------
# ---------------- Patching and Sliding-Window ------------------
# ---------------------------------------------------------------


def get_random_patch(
    dims: Sequence[int], patch_size: Sequence[int], rand_state: Optional[np.random.RandomState] = None
) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_state: a random state object to generate random numbers from

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def iter_patch_slices(
    dims: Sequence[int], patch_size: Union[Sequence[int], int], start_pos: Sequence[int] = ()
) -> Generator[Tuple[slice, ...], None, None]:
    """
    Yield successive tuples of slices defining patches of size `patch_size` from an array of dimensions `dims`. The
    iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a first dimension as least significant ordering.

    Args:
        dims: dimensions of array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension

    Yields:
        Tuples of slice objects defining each patch
    """

    # ensure patchSize and startPos are the right length
    ndim = len(dims)
    patch_size_ = get_valid_patch_size(dims, patch_size)
    start_pos = ensure_tuple_size(start_pos, ndim)

    # collect the ranges to step over each dimension
    ranges = tuple(starmap(range, zip(start_pos, dims, patch_size_)))

    # choose patches by applying product to the ranges
    for position in product(*ranges[::-1]):  # reverse ranges order to iterate in index order
        yield tuple(slice(s, s + p) for s, p in zip(position[::-1], patch_size_))


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
) -> List[Tuple[slice, ...]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    slices = [tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]
    return slices


def iter_patch(
    arr: np.ndarray,
    patch_size: Union[Sequence[int], int] = 0,
    start_pos: Sequence[int] = (),
    copy_back: bool = True,
    mode: Union[NumpyPadMode, str] = NumpyPadMode.WRAP,
    **pad_opts: Dict,
) -> Generator[np.ndarray, None, None]:
    """
    Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
    but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
    to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

    Args:
        arr: array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        copy_back: if True data from the yielded patches is copied back to `arr` once the generator completes
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        pad_opts: padding options, see `numpy.pad`

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes.
    """
    # ensure patchSize and startPos are the right length
    patch_size_ = get_valid_patch_size(arr.shape, patch_size)
    start_pos = ensure_tuple_size(start_pos, arr.ndim)

    # pad image by maximum values needed to ensure patches are taken from inside an image
    arrpad = np.pad(arr, tuple((p, p) for p in patch_size_), NumpyPadMode(mode).value, **pad_opts)

    # choose a start position in the padded image
    start_pos_padded = tuple(s + p for s, p in zip(start_pos, patch_size_))

    # choose a size to iterate over which is smaller than the actual padded image to prevent producing
    # patches which are only in the padded regions
    iter_size = tuple(s + p for s, p in zip(arr.shape, patch_size_))

    for slices in iter_patch_slices(iter_size, patch_size_, start_pos_padded):
        yield arrpad[slices]

    # copy back data from the padded image if required
    if copy_back:
        slices = tuple(slice(p, p + s) for p, s in zip(patch_size_, arr.shape))
        arr[...] = arrpad[slices]


def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def compute_importance_map(
    patch_size: Tuple[int, ...],
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    device: Union[torch.device, int, str] = "cpu",
) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = BlendMode(mode)
    device = torch.device(device)  # type: ignore[arg-type]
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device).float()
    elif mode == BlendMode.GAUSSIAN:
        center_coords = [i // 2 for i in patch_size]
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        importance_map = torch.zeros(patch_size, device=device)
        importance_map[tuple(center_coords)] = 1
        pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(device=device, dtype=torch.float)
        importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
        importance_map = importance_map.squeeze(0).squeeze(0)
        importance_map = importance_map / torch.max(importance_map)
        importance_map = importance_map.float()

        # importance_map cannot be 0, otherwise we may end up with nans!
        min_non_zero = importance_map[importance_map != 0].min().item()
        importance_map = torch.clamp(importance_map, min=min_non_zero)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )

    return importance_map


def same_padding(
    kernel_size: Union[Sequence[int], int], dilation: Union[Sequence[int], int] = 1
) -> Union[Tuple[int, ...], int]:
    """
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.

    Raises:
        NotImplementedError: When ``np.any((kernel_size - 1) * dilation % 2 == 1)``.

    """

    kernel_size_np = np.atleast_1d(kernel_size)
    dilation_np = np.atleast_1d(dilation)

    if np.any((kernel_size_np - 1) * dilation % 2 == 1):
        raise NotImplementedError(
            f"Same padding not available for kernel_size={kernel_size_np} and dilation={dilation_np}."
        )

    padding_np = (kernel_size_np - 1) / 2 * dilation_np
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


class GaussianFilter(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor],
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
                must have shape (Batch, channels, H[, W, ...]).
            sigma: std. could be a single value, or `spatial_dims` number of values.
            truncated: spreads how many stds.
            approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

                - ``erf`` approximation interpolates the error function;
                - ``sampled`` uses a sampled Gaussian kernel;
                - ``scalespace`` corresponds to
                  https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
                  based on the modified Bessel functions.

            requires_grad: whether to store the gradients for sigma.
                if True, `sigma` will be the initial value of the parameters of this module
                (for example `parameters()` iterator could be used to get the parameters);
                otherwise this module will fix the kernels using `sigma` as the std.
        """
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(s, dtype=torch.float, device=s.device if torch.is_tensor(s) else None),
                requires_grad=requires_grad,
            )
            for s in ensure_tuple_rep(sigma, int(spatial_dims))
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        _kernel = [gaussian_1d(s, truncated=self.truncated, approx=self.approx) for s in self.sigma]
        return separable_filtering(x=x, kernels=_kernel)


def separable_filtering(x: torch.Tensor, kernels: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Apply 1-D convolutions along each spatial dimension of `x`.

    Args:
        x: the input image. must have shape (batch, channels, H[, W, ...]).
        kernels: kernel along each spatial dimension.
            could be a single kernel (duplicated for all dimension), or `spatial_dims` number of kernels.

    Raises:
        TypeError: When ``x`` is not a ``torch.Tensor``.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    _kernels = [
        torch.as_tensor(s, dtype=torch.float, device=s.device if torch.is_tensor(s) else None)
        for s in ensure_tuple_rep(kernels, spatial_dims)
    ]
    _paddings = [cast(int, (same_padding(k.shape[0]))) for k in _kernels]
    n_chns = x.shape[1]

    def _conv(input_: torch.Tensor, d: int) -> torch.Tensor:
        if d < 0:
            return input_
        s = [1] * len(input_.shape)
        s[d + 2] = -1
        _kernel = kernels[d].reshape(s)
        _kernel = _kernel.repeat([n_chns, 1] + [1] * spatial_dims)
        _padding = [0] * spatial_dims
        _padding[d] = _paddings[d]
        conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
        return conv_type(input=_conv(input_, d - 1), weight=_kernel, padding=_padding, groups=n_chns)

    return _conv(x, spatial_dims - 1)


def polyval(coef, x) -> torch.Tensor:
    """
    Evaluates the polynomial defined by `coef` at `x`.

    For a 1D sequence of coef (length n), evaluate::

        y = coef[n-1] + x * (coef[n-2] + ... + x * (coef[1] + x * coef[0]))

    Args:
        coef: a sequence of floats representing the coefficients of the polynomial
        x: float or a sequence of floats representing the variable of the polynomial

    Returns:
        1D torch tensor
    """
    device = x.device if torch.is_tensor(x) else None
    coef = torch.as_tensor(coef, dtype=torch.float, device=device)
    if coef.ndim == 0 or (len(coef) < 1):
        return torch.zeros(x.shape)
    x = torch.as_tensor(x, dtype=torch.float, device=device)
    ans = coef[0]
    for c in coef[1:]:
        ans = ans * x + c
    return ans  # type: ignore


def _modified_bessel_0(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        return polyval([0.45813e-2, 0.360768e-1, 0.2659732, 1.2067492, 3.0899424, 3.5156229, 1.0], y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        0.392377e-2,
        -0.1647633e-1,
        0.2635537e-1,
        -0.2057706e-1,
        0.916281e-2,
        -0.157565e-2,
        0.225319e-2,
        0.1328592e-1,
        0.39894228,
    ]
    return polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)


def _modified_bessel_1(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
    if torch.abs(x) < 3.75:
        y = x * x / 14.0625
        _coef = [0.32411e-3, 0.301532e-2, 0.2658733e-1, 0.15084934, 0.51498869, 0.87890594, 0.5]
        return torch.abs(x) * polyval(_coef, y)
    ax = torch.abs(x)
    y = 3.75 / ax
    _coef = [
        -0.420059e-2,
        0.1787654e-1,
        -0.2895312e-1,
        0.2282967e-1,
        -0.1031555e-1,
        0.163801e-2,
        -0.362018e-2,
        -0.3988024e-1,
        0.39894228,
    ]
    ans = polyval(_coef, y) * torch.exp(ax) / torch.sqrt(ax)
    return -ans if x < 0.0 else ans


def _modified_bessel_i(n: int, x: torch.Tensor) -> torch.Tensor:
    if n < 2:
        raise ValueError(f"n must be greater than 1, got n={n}.")
    x = torch.as_tensor(x, dtype=torch.float, device=x.device if torch.is_tensor(x) else None)
    if x == 0.0:
        return x
    device = x.device
    tox = 2.0 / torch.abs(x)
    ans, bip, bi = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
    m = int(2 * (n + np.floor(np.sqrt(40.0 * n))))
    for j in range(m, 0, -1):
        bim = bip + float(j) * tox * bi
        bip = bi
        bi = bim
        if abs(bi) > 1.0e10:
            ans = ans * 1.0e-10
            bi = bi * 1.0e-10
            bip = bip * 1.0e-10
        if j == n:
            ans = bip
    ans = ans * _modified_bessel_0(x) / bi
    return -ans if x < 0.0 and (n % 2) == 1 else ans


def gaussian_1d(
    sigma: torch.Tensor, truncated: float = 4.0, approx: str = "erf", normalize: bool = False
) -> torch.Tensor:
    """
    one dimensional Gaussian kernel.

    Args:
        sigma: std of the kernel
        truncated: tail length
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".

            - ``erf`` approximation interpolates the error function;
            - ``sampled`` uses a sampled Gaussian kernel;
            - ``scalespace`` corresponds to
              https://en.wikipedia.org/wiki/Scale_space_implementation#The_discrete_Gaussian_kernel
              based on the modified Bessel functions.

        normalize: whether to normalize the kernel with `kernel.sum()`.

    Raises:
        ValueError: When ``truncated`` is non-positive.

    Returns:
        1D torch tensor

    """
    sigma = torch.as_tensor(sigma, dtype=torch.float, device=sigma.device if torch.is_tensor(sigma) else None)
    device = sigma.device
    if truncated <= 0.0:
        raise ValueError(f"truncated must be positive, got {truncated}.")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=sigma.device)
        out = torch.exp(-0.5 / (sigma * sigma) * x ** 2)
        if not normalize:  # compute the normalizer
            out = out / (2.5066282 * sigma)
    elif approx.lower() == "scalespace":
        sigma2 = sigma * sigma
        out_pos: List[Optional[torch.Tensor]] = [None] * (tail + 1)
        out_pos[0] = _modified_bessel_0(sigma2)
        out_pos[1] = _modified_bessel_1(sigma2)
        for k in range(2, len(out_pos)):
            out_pos[k] = _modified_bessel_i(k, sigma2)
        out = out_pos[:0:-1]
        out.extend(out_pos)
        out = torch.stack(out) * torch.exp(-sigma2)
    else:
        raise NotImplementedError(f"Unsupported option: approx='{approx}'.")
    return out / out.sum() if normalize else out  # type: ignore


