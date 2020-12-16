#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""
****** NOTE: ALL THE CODE BELOW ARE TAKEN FROM TORCHIO WITH MODIFICATION******
            https://github.com/fepegar/torchio
"""

from typing import Tuple, Optional, Union, Sequence, Dict
import torch
import numpy as np
import numbers
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Sequence, Iterable, List
import nibabel as nib
import scipy.ndimage as ndi

TypeTripletInt = Tuple[int, int, int]
TypeTuple = Union[int, TypeTripletInt]
TypeTripletInt = Tuple[int, int, int]
TypeLocations = Sequence[Tuple[TypeTripletInt, TypeTripletInt]]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTripletFloat = Tuple[float, float, float]
TypeNumber = Union[int, float]
TypeTransformInput = Union[
    torch.Tensor,
    np.ndarray,
    dict,
    nib.Nifti1Image,
]
TypeSextetFloat = Tuple[float, float, float, float, float, float]

# class Transform(ABC):
#     """Abstract class for all TorchIO transforms.
#
#     All subclasses should overwrite
#     :meth:`torchio.tranforms.Transform.apply_transform`,
#     which takes data, applies some transformation and returns the result.
#
#     The input can be an instance of
#     :class:`torchio.Subject`,
#     :class:`torchio.Image`,
#     :class:`numpy.ndarray`,
#     :class:`torch.Tensor`,
#     :class:`SimpleITK.image`,
#     or a Python dictionary.
#
#     Args:
#         p: Probability that this transform will be applied.
#         copy: Make a shallow copy of the input before applying the transform.
#         keys: Mandatory if the input is a Python dictionary. The transform will
#             be applied only to the data in each key.
#     """
#     def __init__(
#             self,
#             p: float = 1,
#             copy: bool = True,
#             keys: Optional[Sequence[str]] = None,
#             ):
#         self.probability = self.parse_probability(p)
#         self.copy = copy
#         self.keys = keys
#
#     def __call__(
#             self,
#             data: TypeTransformInput,
#             ) -> TypeTransformInput:
#         """Transform data and return a result of the same type.
#
#         Args:
#             data: Instance of 1) :class:`~torchio.Subject`, 4D
#                 :class:`torch.Tensor` or NumPy array with dimensions
#                 :math:`(C, W, H, D)`, where :math:`C` is the number of channels
#                 and :math:`W, H, D` are the spatial dimensions. If the input is
#                 a tensor, the affine matrix will be set to identity. Other
#                 valid input types are a SimpleITK image, a
#                 :class:`torch.Image`, a NiBabel Nifti1 Image or a Python
#                 dictionary. The output type is the same as te input type.
#         """
#         if torch.rand(1).item() > self.probability:
#             return data
#         data_parser = DataParser(data, keys=self.keys)
#         subject = data_parser.get_subject()
#         if self.copy:
#             subject = copy.copy(subject)
#         with np.errstate(all='raise'):
#             transformed = self.apply_transform(subject)
#         self.add_transform_to_subject_history(transformed)
#         for image in transformed.get_images(intensity_only=False):
#             ndim = image.data.ndim
#             assert ndim == 4, f'Output of {self.name} is {ndim}D'
#             dtype = image.data.dtype
#             assert dtype is torch.float32, f'Output of {self.name} is {dtype}'
#
#         output = data_parser.get_output(transformed)
#         return output
#
#     def __repr__(self):
#         if hasattr(self, 'args_names'):
#             names = self.args_names
#             args_strings = [f'{arg}={getattr(self, arg)}' for arg in names]
#             if hasattr(self, 'invert_transform') and self.invert_transform:
#                 args_strings.append('invert=True')
#             args_string = ', '.join(args_strings)
#             return f'{self.name}({args_string})'
#         else:
#             return super().__repr__()
#
#     @property
#     def name(self):
#         return self.__class__.__name__
#
#     @abstractmethod
#     def apply_transform(self, subject: Subject):
#         raise NotImplementedError
#
#     def add_transform_to_subject_history(self, subject):
#         from .augmentation import RandomTransform
#         from . import Compose, OneOf, CropOrPad
#         call_others = (
#             RandomTransform,
#             Compose,
#             OneOf,
#             CropOrPad,
#         )
#         if not isinstance(self, call_others):
#             subject.add_transform(self, self._get_reproducing_arguments())
#
#     @staticmethod
#     def to_range(n, around):
#         if around is None:
#             return 0, n
#         else:
#             return around - n, around + n
#
#     def parse_params(self, params, around, name, make_ranges=True, **kwargs):
#         params = to_tuple(params)
#         if len(params) == 1 or (len(params) == 2 and make_ranges):  # d or (a, b)
#             params *= 3  # (d, d, d) or (a, b, a, b, a, b)
#         if len(params) == 3 and make_ranges:  # (a, b, c)
#             items = [self.to_range(n, around) for n in params]
#             # (-a, a, -b, b, -c, c) or (1-a, 1+a, 1-b, 1+b, 1-c, 1+c)
#             params = [n for prange in items for n in prange]
#         if make_ranges:
#             if len(params) != 6:
#                 message = (
#                     f'If "{name}" is a sequence, it must have length 2, 3 or 6,'
#                     f' not {len(params)}'
#                 )
#                 raise ValueError(message)
#             for param_range in zip(params[::2], params[1::2]):
#                 self.parse_range(param_range, name, **kwargs)
#         return tuple(params)
#
#     @staticmethod
#     def parse_range(
#             nums_range: Union[TypeNumber, Tuple[TypeNumber, TypeNumber]],
#             name: str,
#             min_constraint: TypeNumber = None,
#             max_constraint: TypeNumber = None,
#             type_constraint: type = None,
#             ) -> Tuple[TypeNumber, TypeNumber]:
#         r"""Adapted from ``torchvision.transforms.RandomRotation``.
#
#         Args:
#             nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
#                 where :math:`n_{min} \leq n_{max}`.
#                 If a single positive number :math:`n` is provided,
#                 :math:`n_{min} = -n` and :math:`n_{max} = n`.
#             name: Name of the parameter, so that an informative error message
#                 can be printed.
#             min_constraint: Minimal value that :math:`n_{min}` can take,
#                 default is None, i.e. there is no minimal value.
#             max_constraint: Maximal value that :math:`n_{max}` can take,
#                 default is None, i.e. there is no maximal value.
#             type_constraint: Precise type that :math:`n_{max}` and
#                 :math:`n_{min}` must take.
#
#         Returns:
#             A tuple of two numbers :math:`(n_{min}, n_{max})`.
#
#         Raises:
#             ValueError: if :attr:`nums_range` is negative
#             ValueError: if :math:`n_{max}` or :math:`n_{min}` is not a number
#             ValueError: if :math:`n_{max} \lt n_{min}`
#             ValueError: if :attr:`min_constraint` is not None and
#                 :math:`n_{min}` is smaller than :attr:`min_constraint`
#             ValueError: if :attr:`max_constraint` is not None and
#                 :math:`n_{max}` is greater than :attr:`max_constraint`
#             ValueError: if :attr:`type_constraint` is not None and
#                 :math:`n_{max}` and :math:`n_{max}` are not of type
#                 :attr:`type_constraint`.
#         """
#         if isinstance(nums_range, numbers.Number):  # single number given
#             if nums_range < 0:
#                 raise ValueError(
#                     f'If {name} is a single number,'
#                     f' it must be positive, not {nums_range}')
#             if min_constraint is not None and nums_range < min_constraint:
#                 raise ValueError(
#                     f'If {name} is a single number, it must be greater'
#                     f' than {min_constraint}, not {nums_range}'
#                 )
#             if max_constraint is not None and nums_range > max_constraint:
#                 raise ValueError(
#                     f'If {name} is a single number, it must be smaller'
#                     f' than {max_constraint}, not {nums_range}'
#                 )
#             if type_constraint is not None:
#                 if not isinstance(nums_range, type_constraint):
#                     raise ValueError(
#                         f'If {name} is a single number, it must be of'
#                         f' type {type_constraint}, not {nums_range}'
#                     )
#             min_range = -nums_range if min_constraint is None else nums_range
#             return (min_range, nums_range)
#
#         try:
#             min_value, max_value = nums_range
#         except (TypeError, ValueError):
#             raise ValueError(
#                 f'If {name} is not a single number, it must be'
#                 f' a sequence of len 2, not {nums_range}'
#             )
#
#         min_is_number = isinstance(min_value, numbers.Number)
#         max_is_number = isinstance(max_value, numbers.Number)
#         if not min_is_number or not max_is_number:
#             message = (
#                 f'{name} values must be numbers, not {nums_range}')
#             raise ValueError(message)
#
#         if min_value > max_value:
#             raise ValueError(
#                 f'If {name} is a sequence, the second value must be'
#                 f' equal or greater than the first, but it is {nums_range}')
#
#         if min_constraint is not None and min_value < min_constraint:
#             raise ValueError(
#                 f'If {name} is a sequence, the first value must be greater'
#                 f' than {min_constraint}, but it is {min_value}'
#             )
#
#         if max_constraint is not None and max_value > max_constraint:
#             raise ValueError(
#                 f'If {name} is a sequence, the second value must be smaller'
#                 f' than {max_constraint}, but it is {max_value}'
#             )
#
#         if type_constraint is not None:
#             min_type_ok = isinstance(min_value, type_constraint)
#             max_type_ok = isinstance(max_value, type_constraint)
#             if not min_type_ok or not max_type_ok:
#                 raise ValueError(
#                     f'If "{name}" is a sequence, its values must be of'
#                     f' type "{type_constraint}", not "{type(nums_range)}"'
#                 )
#         return nums_range
#
#     @staticmethod
#     def parse_interpolation(interpolation: str) -> str:
#         if not isinstance(interpolation, str):
#             itype = type(interpolation)
#             raise TypeError(f'Interpolation must be a string, not {itype}')
#         interpolation = interpolation.lower()
#         is_string = isinstance(interpolation, str)
#         supported_values = [key.name.lower() for key in Interpolation]
#         is_supported = interpolation.lower() in supported_values
#         if is_string and is_supported:
#             return interpolation
#         message = (
#             f'Interpolation "{interpolation}" of type {type(interpolation)}'
#             f' must be a string among the supported values: {supported_values}'
#         )
#         raise ValueError(message)
#
#     @staticmethod
#     def parse_probability(probability: float) -> float:
#         is_number = isinstance(probability, numbers.Number)
#         if not (is_number and 0 <= probability <= 1):
#             message = (
#                 'Probability must be a number in [0, 1],'
#                 f' not {probability}'
#             )
#             raise ValueError(message)
#         return probability
#
#     @staticmethod
#     def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
#         return nib_to_sitk(data, affine)
#
#     @staticmethod
#     def sitk_to_nib(image: sitk.Image) -> Tuple[torch.Tensor, np.ndarray]:
#         return sitk_to_nib(image)
#
#     def _get_reproducing_arguments(self):
#         """
#         Return a dictionary with the arguments that would be necessary to
#         reproduce the transform exactly.
#         """
#         return {name: getattr(self, name) for name in self.args_names}
#
#     def is_invertible(self):
#         return hasattr(self, 'invert_transform')
#
#     def inverse(self):
#         if not self.is_invertible():
#             raise RuntimeError(f'{self.name} is not invertible')
#         new = copy.deepcopy(self)
#         new.invert_transform = not self.invert_transform
#         return new
#
#     @staticmethod
#     @contextmanager
#     def _use_seed(seed):
#         """Perform an operation using a specific seed for the PyTorch RNG"""
#         torch_rng_state = torch.random.get_rng_state()
#         torch.manual_seed(seed)
#         yield
#         torch.random.set_rng_state(torch_rng_state)
#
#     @staticmethod
#     def get_sitk_interpolator(interpolation: str) -> int:
#         return get_sitk_interpolator(interpolation)
#
#
# class IntensityTransform(Transform):
#     """Transform that modifies voxel intensities only."""
#     @staticmethod
#     def get_images_dict(sample):
#         return sample.get_images_dict(intensity_only=True)
#
#     def arguments_are_dict(self) -> bool:
#         """Check if main arguments are dict.
#
#         Return True if the type of all attributes specified in the
#         :attr:`args_names` have ``dict`` type.
#         """
#         args = [getattr(self, name) for name in self.args_names]
#         are_dict = [isinstance(arg, dict) for arg in args]
#         if all(are_dict):
#             return True
#         elif not any(are_dict):
#             return False
#         else:
#             message = 'Either all or none of the arguments must be dicts'
#             raise ValueError(message)
#
#
# class RandomTransform(Transform):
#     """Base class for stochastic augmentation transforms.
#
#     Args:
#         p: Probability that this transform will be applied.
#         keys: See :class:`~torchio.transforms.Transform`.
#     """
#     def __init__(
#             self,
#             p: float = 1,
#             keys: Optional[Sequence[str]] = None,
#             ):
#         super().__init__(p=p, keys=keys)
#
#     def parse_degrees(
#             self,
#             degrees: TypeRangeFloat,
#             ) -> Tuple[float, float]:
#         return self.parse_range(degrees, 'degrees')
#
#     def parse_translation(
#             self,
#             translation: TypeRangeFloat,
#             ) -> Tuple[float, float]:
#         return self.parse_range(translation, 'translation')
#
#     @staticmethod
#     def sample_uniform(a, b):
#         return torch.FloatTensor(1).uniform_(a, b)
#
#     @staticmethod
#     def get_random_seed():
#         """Generate a random seed.
#
#         Returns:
#             A random seed as an int.
#         """
#         return torch.randint(0, 2**31, (1,)).item()
#
#     def sample_uniform_sextet(self, params):
#         results = []
#         for (a, b) in zip(params[::2], params[1::2]):
#             results.append(self.sample_uniform(a, b))
#         return torch.Tensor(results)


def parse_range(
        nums_range: Union[TypeNumber, Tuple[TypeNumber, TypeNumber]],
        name: str,
        min_constraint: TypeNumber = None,
        max_constraint: TypeNumber = None,
        type_constraint: type = None,
) -> Tuple[TypeNumber, TypeNumber]:
    r"""Adapted from ``torchvision.transforms.RandomRotation``.

    Args:
        nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
            where :math:`n_{min} \leq n_{max}`.
            If a single positive number :math:`n` is provided,
            :math:`n_{min} = -n` and :math:`n_{max} = n`.
        name: Name of the parameter, so that an informative error message
            can be printed.
        min_constraint: Minimal value that :math:`n_{min}` can take,
            default is None, i.e. there is no minimal value.
        max_constraint: Maximal value that :math:`n_{max}` can take,
            default is None, i.e. there is no maximal value.
        type_constraint: Precise type that :math:`n_{max}` and
            :math:`n_{min}` must take.

    Returns:
        A tuple of two numbers :math:`(n_{min}, n_{max})`.

    Raises:
        ValueError: if :attr:`nums_range` is negative
        ValueError: if :math:`n_{max}` or :math:`n_{min}` is not a number
        ValueError: if :math:`n_{max} \lt n_{min}`
        ValueError: if :attr:`min_constraint` is not None and
            :math:`n_{min}` is smaller than :attr:`min_constraint`
        ValueError: if :attr:`max_constraint` is not None and
            :math:`n_{max}` is greater than :attr:`max_constraint`
        ValueError: if :attr:`type_constraint` is not None and
            :math:`n_{max}` and :math:`n_{max}` are not of type
            :attr:`type_constraint`.
    """
    if isinstance(nums_range, numbers.Number):  # single number given
        if nums_range < 0:
            raise ValueError(
                f'If {name} is a single number,'
                f' it must be positive, not {nums_range}')
        if min_constraint is not None and nums_range < min_constraint:
            raise ValueError(
                f'If {name} is a single number, it must be greater'
                f' than {min_constraint}, not {nums_range}'
            )
        if max_constraint is not None and nums_range > max_constraint:
            raise ValueError(
                f'If {name} is a single number, it must be smaller'
                f' than {max_constraint}, not {nums_range}'
            )
        if type_constraint is not None:
            if not isinstance(nums_range, type_constraint):
                raise ValueError(
                    f'If {name} is a single number, it must be of'
                    f' type {type_constraint}, not {nums_range}'
                )
        min_range = -nums_range if min_constraint is None else nums_range
        return (min_range, nums_range)

    try:
        min_value, max_value = nums_range
    except (TypeError, ValueError):
        raise ValueError(
            f'If {name} is not a single number, it must be'
            f' a sequence of len 2, not {nums_range}'
        )

    min_is_number = isinstance(min_value, numbers.Number)
    max_is_number = isinstance(max_value, numbers.Number)
    if not min_is_number or not max_is_number:
        message = (
            f'{name} values must be numbers, not {nums_range}')
        raise ValueError(message)

    if min_value > max_value:
        raise ValueError(
            f'If {name} is a sequence, the second value must be'
            f' equal or greater than the first, but it is {nums_range}')

    if min_constraint is not None and min_value < min_constraint:
        raise ValueError(
            f'If {name} is a sequence, the first value must be greater'
            f' than {min_constraint}, but it is {min_value}'
        )

    if max_constraint is not None and max_value > max_constraint:
        raise ValueError(
            f'If {name} is a sequence, the second value must be smaller'
            f' than {max_constraint}, but it is {max_value}'
        )

    if type_constraint is not None:
        min_type_ok = isinstance(min_value, type_constraint)
        max_type_ok = isinstance(max_value, type_constraint)
        if not min_type_ok or not max_type_ok:
            raise ValueError(
                f'If "{name}" is a sequence, its values must be of'
                f' type "{type_constraint}", not "{type(nums_range)}"'
            )
    return nums_range


def to_tuple(
        value: Union[TypeNumber, Iterable[TypeNumber]],
        length: int = 1,
        ) -> Union[TypeTripletFloat, Tuple[TypeNumber, ...]]:
    """
    to_tuple(1, length=1) -> (1,)
    to_tuple(1, length=3) -> (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned
    to_tuple((1,), length=1) -> (1,)
    to_tuple((1, 2), length=1) -> (1, 2)
    to_tuple([1, 2], length=3) -> (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value


def to_range(n, around):
    if around is None:
        return 0, n
    else:
        return around - n, around + n


def parse_params(params, around, name, make_ranges=True, **kwargs):
    params = to_tuple(params)
    if len(params) == 1 or (len(params) == 2 and make_ranges):  # d or (a, b)
        params *= 3  # (d, d, d) or (a, b, a, b, a, b)
    if len(params) == 3 and make_ranges:  # (a, b, c)
        items = [to_range(n, around) for n in params]
        # (-a, a, -b, b, -c, c) or (1-a, 1+a, 1-b, 1+b, 1-c, 1+c)
        params = [n for prange in items for n in prange]
    if make_ranges:
        if len(params) != 6:
            message = (
                f'If "{name}" is a sequence, it must have length 2, 3 or 6,'
                f' not {len(params)}'
            )
            raise ValueError(message)
        for param_range in zip(params[::2], params[1::2]):
            parse_range(param_range, name, **kwargs)
    return tuple(params)


def sample_uniform(a, b):
    return torch.FloatTensor(1).uniform_(a, b)


def sample_uniform_sextet(self, params):
    results = []
    for (a, b) in zip(params[::2], params[1::2]):
        results.append(self.sample_uniform(a, b))
    return torch.Tensor(results)


class Transformable(ABC):
    def __call__(self, volume):
        return self.apply_transform(volume)

    @abstractmethod
    def apply_transform(self, volume):
        raise NotImplementedError


class Randomizeable(ABC):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, volume):
        if torch.rand(1).item() > self.p:
            return volume
        return self.apply_transform(volume)

    @abstractmethod
    def apply_transform(self, volume):
        raise NotImplementedError


class FourierTransform:

    @staticmethod
    def fourier_transform(array: np.ndarray) -> np.ndarray:
        transformed = np.fft.fftn(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift: np.ndarray) -> np.ndarray:
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifftn(f_ishift)
        return img_back


class RandomSpike(Randomizeable, FourierTransform):
    r"""Add random MRI spike artifacts.

    Also known as `Herringbone artifact
    <https://radiopaedia.org/articles/herringbone-artifact?lang=gb>`_,
    crisscross artifact or corduroy artifact, it creates stripes in different
    directions in image space due to spikes in k-space.

    Args:
        num_spikes: Number of spikes :math:`n` present in k-space.
            If a tuple :math:`(a, b)` is provided, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :math:`d` is provided,
            :math:`n \sim \mathcal{U}(0, d) \cap \mathbb{N}`.
            Larger values generate more distorted images.
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
            If a tuple :math:`(a, b)` is provided, then
            :math:`r \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`r \sim \mathcal{U}(-d, d)`.
            Larger values generate more distorted images.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            num_spikes: Union[int, Tuple[int, int]] = 1,
            intensity: Union[float, Tuple[float, float]] = (1, 3),
            p: float = 1,
            ):
        super().__init__(p)
        self.intensity_range = parse_range(
            intensity, 'intensity_range')
        self.num_spikes_range = parse_range(
            num_spikes, 'num_spikes', min_constraint=0, type_constraint=int)

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        arguments = dict()
        spikes_positions_param, intensity_param = self.get_params(
            self.num_spikes_range,
            self.intensity_range,
        )
        arguments['spikes_positions'] = spikes_positions_param
        arguments['intensity'] = intensity_param
        transform = Spike(**arguments)
        volume = transform(volume)
        return volume

    @staticmethod
    def get_params(
            num_spikes_range: Tuple[int, int],
            intensity_range: Tuple[float, float],
            ) -> Tuple[np.ndarray, float]:
        ns_min, ns_max = num_spikes_range
        num_spikes_param = torch.randint(ns_min, ns_max + 1, (1,)).item()
        intensity_param = sample_uniform(*intensity_range)
        spikes_positions = torch.rand(num_spikes_param, 3).numpy()
        return spikes_positions, intensity_param.item()


class Spike(Transformable, FourierTransform):
    r"""Add MRI spike artifacts.

    Also known as `Herringbone artifact
    <https://radiopaedia.org/articles/herringbone-artifact?lang=gb>`_,
    crisscross artifact or corduroy artifact, it creates stripes in different
    directions in image space due to spikes in k-space.

    Args:
        spikes_positions:
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
        keys: See :class:`~torchio.transforms.Transform`.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            spikes_positions: Union[np.ndarray, Dict[str, np.ndarray]],
            intensity: Union[float, Dict[str, float]],
            keys: Optional[Sequence[str]] = None,
            ):
        self.spikes_positions = spikes_positions
        self.intensity = intensity
        self.args_names = 'spikes_positions', 'intensity'
        self.invert_transform = False

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        spikes_positions = self.spikes_positions
        intensity = self.intensity
        transformed_tensors = []
        for channel in volume:
            transformed_tensor = self.add_artifact(
                channel,
                spikes_positions,
                intensity,
            )
            transformed_tensors.append(transformed_tensor)
        volume = torch.stack(transformed_tensors)
        return volume

    def add_artifact(
            self,
            tensor: torch.Tensor,
            spikes_positions: np.ndarray,
            intensity_factor: float,
            ):
        array = np.asarray(tensor)
        spectrum = self.fourier_transform(array)
        shape = np.array(spectrum.shape)
        mid_shape = shape // 2
        indices = np.floor(spikes_positions * shape).astype(int)
        for index in indices:
            diff = index - mid_shape
            i, j, k = mid_shape + diff
            artifact = spectrum.max() * intensity_factor
            if self.invert_transform:
                spectrum[i, j, k] -= artifact
            else:
                spectrum[i, j, k] += artifact
            # If we wanted to add a pure cosine, we should add spikes to both
            # sides of k-space. However, having only one is a better
            # representation og the actual cause of the artifact in real
            # scans. Therefore the next two lines have been removed.
            # #i, j, k = mid_shape - diff
            # #spectrum[i, j, k] = spectrum.max() * intensity_factor
        result = np.real(self.inv_fourier_transform(spectrum))
        return torch.from_numpy(result.astype(np.float32))


class RandomGhosting(Randomizeable):
    r"""Add random MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com <http://mriquestions.com/why-discrete-ghosts.html>`_).

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
            If :attr:`num_ghosts` is a tuple :math:`(a, b)`, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :math:`d` is provided,
            :math:`n \sim \mathcal{U}(0, d) \cap \mathbb{N}`.
        axes: Axis along which the ghosts will be created. If
            :attr:`axes` is a tuple, the axis will be randomly chosen
            from the passed values. Anatomical labels may also be used (see
            :class:`~torchio.transforms.augmentation.RandomFlip`).
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible. If a tuple
            :math:`(a, b)` is provided then :math:`s \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`s \sim \mathcal{U}(0, d)`.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Tuple[int, int]] = (4, 10),
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            intensity: Union[float, Tuple[float, float]] = (0.5, 1),
            restore: float = 0.02,
            p: float = 1,
            ):
        super().__init__(p)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)
            except TypeError:
                axes = (axes,)
        for axis in axes:
            if not isinstance(axis, str) and axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        self.num_ghosts_range = parse_range(
            num_ghosts, 'num_ghosts', min_constraint=0, type_constraint=int)
        self.intensity_range = parse_range(
            intensity, 'intensity_range', min_constraint=0)
        self.restore = self._parse_restore(restore)

    @staticmethod
    def _parse_restore(restore):
        if not isinstance(restore, float):
            raise TypeError(f'Restore must be a float, not {restore}')
        if not 0 <= restore <= 1:
            message = (
                f'Restore must be a number between 0 and 1, not {restore}')
            raise ValueError(message)
        return restore

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        arguments = dict()
        # if any(isinstance(n, str) for n in self.axes):
        #     subject.check_consistent_orientation()
        # is_2d = image.is_2d()
        axes = self.axes
        num_ghosts_param, axis_param, intensity_param = self.get_params(
            self.num_ghosts_range,
            axes,
            self.intensity_range,
        )
        arguments['num_ghosts'] = num_ghosts_param
        arguments['axis'] = axis_param
        arguments['intensity'] = intensity_param
        arguments['restore'] = self.restore
        transform = Ghosting(**arguments)
        volume = transform(volume)
        return volume

    def get_params(
            self,
            num_ghosts_range: Tuple[int, int],
            axes: Tuple[int, ...],
            intensity_range: Tuple[float, float],
            ) -> Tuple:
        ng_min, ng_max = num_ghosts_range
        num_ghosts = torch.randint(ng_min, ng_max + 1, (1,)).item()
        axis = axes[torch.randint(0, len(axes), (1,))]
        intensity = self.sample_uniform(*intensity_range).item()
        return num_ghosts, axis, intensity


class Ghosting(Transformable, FourierTransform):
    r"""Add MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com <http://mriquestions.com/why-discrete-ghosts.html>`_).

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
        axes: Axis along which the ghosts will be created.
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Dict[str, int]],
            axis: Union[int, Dict[str, int]],
            intensity: Union[float, Dict[str, float]],
            restore: Union[float, Dict[str, float]],
            ):
        self.axis = axis
        self.num_ghosts = num_ghosts
        self.intensity = intensity
        self.restore = restore
        self.args_names = 'num_ghosts', 'axis', 'intensity', 'restore'

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        axis = self.axis
        num_ghosts = self.num_ghosts
        intensity = self.intensity
        restore = self.restore
        transformed_tensors = []
        for tensor in volume:
            transformed_tensor = self.add_artifact(
                tensor,
                num_ghosts,
                axis,
                intensity,
                restore,
            )
            transformed_tensors.append(transformed_tensor)
        volume = torch.stack(transformed_tensors)
        return volume

    def add_artifact(
            self,
            tensor: torch.Tensor,
            num_ghosts: int,
            axis: int,
            intensity: float,
            restore_center: float,
            ):
        if not num_ghosts or not intensity:
            return tensor

        array = tensor.numpy()
        spectrum = self.fourier_transform(array)

        shape = np.array(array.shape)
        ri, rj, rk = np.round(restore_center * shape).astype(np.uint16)
        mi, mj, mk = np.array(array.shape) // 2

        # Variable "planes" is the part of the spectrum that will be modified
        if axis == 0:
            planes = spectrum[::num_ghosts, :, :]
            restore = spectrum[mi, :, :].copy()
        elif axis == 1:
            planes = spectrum[:, ::num_ghosts, :]
            restore = spectrum[:, mj, :].copy()
        elif axis == 2:
            planes = spectrum[:, :, ::num_ghosts]
            restore = spectrum[:, :, mk].copy()
        else:
            raise NotImplementedError
        
        # Multiply by 0 if intensity is 1
        planes *= 1 - intensity

        # Restore the center of k-space to avoid extreme artifacts
        if axis == 0:
            spectrum[mi, :, :] = restore
        elif axis == 1:
            spectrum[:, mj, :] = restore
        elif axis == 2:
            spectrum[:, :, mk] = restore

        array_ghosts = self.inv_fourier_transform(spectrum)
        array_ghosts = np.real(array_ghosts).astype(np.float32)
        return torch.from_numpy(array_ghosts)


class RandomBlur(Transformable):
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            std: Union[float, Tuple[float, float]] = (0, 2),
            p: float = 1,
            ):
        super().__init__(p)
        self.std_ranges = parse_params(std, None, 'std', min_constraint=0)

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        arguments = dict()
        stds = [self.get_params(self.std_ranges) for _ in volume]
        arguments['std'] = stds
        transform = Blur(**arguments)
        volume = transform(volume)
        return volume

    @staticmethod
    def get_params(std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = sample_uniform_sextet(std_ranges)
        return std


class Blur(Transformable):
    r"""Blur an image using a Gaussian filter.

    Args:
        std: Tuple :math:`(\sigma_1, \sigma_2, \sigma_3)` representing the
            the standard deviations (in mm) of the standard deviations
            of the Gaussian kernels used to blur the image along each axis.
        spacing: the volume spacing of voxels; default is 1.
    """
    def __init__(
            self,
            std: Union[TypeTripletFloat, Dict[str, TypeTripletFloat], int],
            spacing: Union[TypeTripletFloat, Dict[str, TypeTripletFloat], int] = 1,
            ):
        self.std = std
        self.spacing = spacing

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        std = self.std
        spacing = self.spacing
        stds = to_tuple(std, length=len(volume))
        spacing = to_tuple(spacing, length=len(volume))
        transformed_tensors = []
        for std, spc, tensor in zip(stds, spacing, volume):
            transformed_tensor = self.blur(
                tensor,
                spc,
                std,
            )
            transformed_tensors.append(transformed_tensor)
        volume = torch.stack(transformed_tensors)
        return volume

    @staticmethod
    def blur(
            data: torch.tensor,
            spacing: Union[int, float],
            std_voxel: Union[int, float],
            ) -> torch.Tensor:
        assert data.ndim == 3
        std_physical = np.array(std_voxel) / np.array(spacing)
        blurred = ndi.gaussian_filter(data, std_physical)
        tensor = torch.from_numpy(blurred)
        return tensor


class RandomBiasField(Transformable):
    r"""Add random MRI bias field artifact.

    MRI magnetic field inhomogeneity creates intensity
    variations of very low frequency across the whole image.

    The bias field is modeled as a linear combination of
    polynomial basis functions, as in K. Van Leemput et al., 1999,
    *Automated model-based tissue classification of MR images of the brain*.

    It was implemented in NiftyNet by Carole Sudre and used in
    `Sudre et al., 2017, Longitudinal segmentation of age-related
    white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        coefficients: Maximum magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            coefficients: Union[float, Tuple[float, float]] = 0.5,
            order: int = 3,
            p: float = 1,
            ):
        super().__init__(p)
        self.coefficients_range = parse_range(
            coefficients, 'coefficients_range')
        self.order = self._parse_order(order)

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        arguments = dict()
        coefficients = self.get_params(
            self.order,
            self.coefficients_range,
        )
        arguments['coefficients'] = coefficients
        arguments['order'] = self.order
        transform = BiasField(**arguments)
        volume = transform(volume)
        return volume

    @staticmethod
    def get_params(
            order: int,
            coefficients_range: Tuple[float, float],
            ) -> List[float]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = sample_uniform(*coefficients_range)
                    random_coefficients.append(number.item())
        return random_coefficients

    @staticmethod
    def _parse_order(order):
        if not isinstance(order, int):
            raise TypeError(f'Order must be an int, not {type(order)}')
        if order < 0:
            raise ValueError(f'Order must be a positive int, not {order}')
        return order


class BiasField(Transformable):
    r"""Add MRI bias field artifact.

    Args:
        coefficient: Magnitudes of the polynomial coefficients.
        order: Order of the basis polynomial functions.
    """
    def __init__(
            self,
            coefficient: Union[List[float], Dict[str, List[float]], int],
            order: Union[int, Dict[str, int]],
            ):
        self.order = self._parse_order(order)
        self.coefficients_range = parse_range(coefficient, 'coefficients_range')
        self.invert_transform = False

    def apply_transform(self, volume: torch.tensor, normalize: bool = False) -> torch.tensor:
        coefficients_range, order = self.coefficients_range, self.order
        coefficients = self.get_coefficients(self.order, coefficients_range)
        bias_field = self.generate_bias_field(volume, order, coefficients)
        if self.invert_transform:
            np.divide(1, bias_field, out=bias_field)
        bias_field = torch.from_numpy(bias_field)
        if normalize:
            bias_field -= bias_field.min()
            if bias_field.max() > 0:
                bias_field /= bias_field.max()
        volume = volume * bias_field
        return volume

    @staticmethod
    def generate_bias_field(
            data: TypeData,
            order: int,
            coefficients: Union[TypeData, List],
            ) -> np.ndarray:
        # Create the bias field map using a linear combination of polynomial
        # functions and the coefficients previously sampled
        shape = np.array(data.shape[1:])  # first axis is channels
        half_shape = shape / 2

        ranges = [np.arange(-n, n) for n in half_shape]

        bias_field = np.zeros(shape)
        x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges))

        x_mesh /= x_mesh.max()
        y_mesh /= y_mesh.max()
        z_mesh /= z_mesh.max()

        i = 0
        for x_order in range(order + 1):
            for y_order in range(order + 1 - x_order):
                for z_order in range(order + 1 - (x_order + y_order)):
                    coefficient = coefficients[i]
                    new_map = (
                        coefficient
                        * x_mesh ** x_order
                        * y_mesh ** y_order
                        * z_mesh ** z_order
                    )
                    bias_field += np.transpose(new_map, (1, 0, 2))  # why?
                    i += 1
        bias_field = np.exp(bias_field).astype(np.float32)
        return bias_field

    @staticmethod
    def _parse_order(order):
        if not isinstance(order, int):
            raise TypeError(f'Order must be an int, not {type(order)}')
        if order < 0:
            raise ValueError(f'Order must be a positive int, not {order}')
        return order

    @staticmethod
    def get_coefficients(
            order: int,
            coefficients_range: Union[List[float], float],
    ) -> List[float]:
        # if isinstance(coefficient, list):
        #     return coefficient

        # Setting the appropriate number of coefficients for the creation of the bias field map
        coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = sample_uniform(*coefficients_range)
                    coefficients.append(number.item())

        return coefficients


class RandomSwap(Randomizeable):
    r"""Randomly swap patches within an image.

    This is typically used in `context restoration for self-supervised learning
    <https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_iterations: Number of times that two patches will be swapped.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            patch_size: TypeTuple = 15,
            num_iterations: int = 100,
            p: float = 1,
            ):
        super().__init__(p)
        self.patch_size = np.array(to_tuple(patch_size))
        self.num_iterations = self._parse_num_iterations(num_iterations)

    @staticmethod
    def _parse_num_iterations(num_iterations):
        if not isinstance(num_iterations, int):
            raise TypeError('num_iterations must be an int,'
                            f'not {num_iterations}')
        if num_iterations < 0:
            raise ValueError('num_iterations must be positive,'
                             f'not {num_iterations}')
        return num_iterations

    def get_params(
            self,
            tensor: torch.Tensor,
            patch_size: np.ndarray,
            num_iterations: int,
            ) -> List[Tuple[TypeTripletInt, TypeTripletInt]]:
        spatial_shape = tensor.shape[-3:]
        locations = []
        for _ in range(num_iterations):
            first_ini, first_fin = self.get_random_indices_from_shape(
                spatial_shape,
                patch_size,
            )
            while True:
                second_ini, second_fin = self.get_random_indices_from_shape(
                    spatial_shape,
                    patch_size,
                )
                larger_than_initial = np.all(second_ini >= first_ini)
                less_than_final = np.all(second_fin <= first_fin)
                if larger_than_initial and less_than_final:
                    continue  # patches overlap
                else:
                    break  # patches don't overlap
            location = tuple(first_ini), tuple(second_ini)
            locations.append(location)
        return locations

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        arguments = dict()
        locations = self.get_params(
            volume,
            self.patch_size,
            self.num_iterations,
        )
        arguments['locations'] = locations
        arguments['patch_size'] = self.patch_size
        transform = Swap(**arguments)
        volume = transform(volume)
        return volume

    @staticmethod
    def get_random_indices_from_shape(
            spatial_shape: TypeTripletInt,
            patch_size: TypeTripletInt,
            ) -> Tuple[np.ndarray, np.ndarray]:
        shape_array = np.array(spatial_shape)
        patch_size_array = np.array(patch_size)
        max_index_ini = shape_array - patch_size_array
        if (max_index_ini < 0).any():
            message = (
                f'Patch size {patch_size} cannot be'
                f' larger than image spatial shape {spatial_shape}'
            )
            raise ValueError(message)
        max_index_ini = max_index_ini.astype(np.uint16)
        coordinates = []
        for max_coordinate in max_index_ini.tolist():
            if max_coordinate == 0:
                coordinate = 0
            else:
                coordinate = torch.randint(max_coordinate, size=(1,)).item()
            coordinates.append(coordinate)
        index_ini = np.array(coordinates, np.uint16)
        index_fin = index_ini + patch_size_array
        return index_ini, index_fin


class Swap(Transformable):
    r"""Swap patches within an image.

    This is typically used in `context restoration for self-supervised learning
    <https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_iterations: Number of times that two patches will be swapped.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            patch_size: Union[TypeTripletInt, Dict[str, TypeTripletInt]],
            locations: Union[TypeLocations, Dict[str, TypeLocations]],
            ):
        self.locations = locations
        self.patch_size = patch_size
        self.invert_transform = False

    def apply_transform(self, volume: torch.tensor) -> torch.tensor:
        locations, patch_size = self.locations, self.patch_size
        if self.invert_transform:
            locations.reverse()

        volume = self.swap(volume, patch_size, locations)

        return volume

    def swap(
            self,
            tensor: torch.Tensor,
            patch_size: TypeTuple,
            locations: List[Tuple[np.ndarray, np.ndarray]],
            ) -> torch.tensor:
        tensor = tensor.clone()
        patch_size = np.array(patch_size)
        for first_ini, second_ini in locations:
            first_fin = first_ini + patch_size
            second_fin = second_ini + patch_size
            first_patch = self.crop(tensor, first_ini, first_fin)
            second_patch = self.crop(tensor, second_ini, second_fin).clone()
            self.insert(tensor, first_patch, second_ini)
            self.insert(tensor, second_patch, first_ini)
        return tensor

    @staticmethod
    def insert(tensor: TypeData, patch: TypeData, index_ini: np.ndarray) -> None:
        index_fin = index_ini + np.array(patch.shape[-3:])
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = patch

    @staticmethod
    def crop(
            image: Union[np.ndarray, torch.Tensor],
            index_ini: np.ndarray,
            index_fin: np.ndarray,
            ) -> Union[np.ndarray, torch.Tensor]:
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        return image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
