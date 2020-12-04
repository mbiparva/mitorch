#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as torchF
from data.functional_mitorch import _is_tensor_image_volume


def complex_from_split(arr):
    """Return a complex array of shape (...) from a split array of real and imaginary
        parts with shape (..., 2)
    Args:
        arr (ndarray or torch.Tensor): Array or tensor with shape (..., 2) where arr[..., 0] represents the real part
            of the array and arr[..., 1] represents the imaginary part (note that both arr[..., 0]
            and arr[..., 1] are real).
    Returns:
        out (ndarray or torch.Tensor): Complex array or tensor with shape (...).
    """
    assert isinstance(arr, (np.ndarray, torch.Tensor)), 'Input array must be numpy array or torch tensor.'
    assert len(arr.shape) >= 2, 'Input array must have at least 2 dimensions.'
    assert arr.shape[-1] == 2, 'Last dimension of array must have length 2.'
    out = arr[..., 0] + 1j*arr[..., 1]
    return out


def split_from_complex(arr):
    """Return a split array of shape (..., 2) where the final axis of length 2 represents
        the real and imaginary parts of the input complex array with shape (...)
    Args:
        arr (ndarray or torch.Tensor): Complex array or tensor with shape (...).
    Returns:
        out (ndarray or torch.Tensor): Array or tensor with shape (..., 2) where arr[..., 0] represents the real part
            of the array and arr[..., 1] represents the imaginary part (note that both arr[..., 0]
            and arr[..., 1] are real).
    """
    assert isinstance(arr, (np.ndarray, torch.Tensor)), 'Input array must be numpy array or torch tensor.'
    if isinstance(arr, np.ndarray):
        out = np.zeros(shape=arr.shape + (2,))
        out[..., 0] = np.real(arr)
        out[..., 1] = np.imag(arr)
        return out
    else:
        out = torch.zeros(size=arr.shape + (2,))
        out[..., 0] = torch.real(arr)
        out[..., 1] = torch.imag(arr)
        return out


def ks_motion_fftshift(x, axes=None):
    """Wrapper for np.fft.fftshift that works for arrays or torch.Tensor objects. See description
        from numpy documentation below:
    
        Shift the zero-frequency component to the center of the spectrum.
        
        This function swaps half-spaces for all axes listed (defaults to all). Note that y[0] is
        the Nyquist component only if len(x) is even.
    Args:
        x (ndarray or torch.Tensor): Input array.
        axes (int or shape tuple, optional): Axes over which to shift. Default is None, which shifts all axes.
    Returns:
        out (ndarray or torch.Tensor): The shifted array.
    """
    assert isinstance(x, (np.ndarray, torch.Tensor)), 'Input must be numpy array or torch tensor.'
    is_tensor = False
    if isinstance(x, torch.Tensor):
        is_tensor = True
        x = x.numpy()
    out = np.fft.fftshift(x, axes)
    if is_tensor:
        return torch.Tensor(out)
    else:
        return out


def ks_motion_ifftshift(x, axes=None):
    """Wrapper for np.fft.ifftshift that works for arrays or torch.Tensor objects. See description
        from numpy documentation below:
    
        The inverse of fftshift. Although identical for even-length x, the functions differ by one sample for
        odd-length x.
    Args:
        x (ndarray or torch.Tensor): Input array.
        axes (int or shape tuple, optional): Axes over which to calculate. Default is None, which shifts all axes.
    Returns:
        out (ndarray or torch.Tensor): The shifted array.
    """
    assert isinstance(x, (np.ndarray, torch.Tensor)), 'Input must be numpy array or torch tensor.'
    is_tensor = False
    if isinstance(x, torch.Tensor):
        is_tensor = True
        x = x.numpy()
    out = np.fft.ifftshift(x, axes)
    if is_tensor:
        return torch.Tensor(out)
    else:
        return out


def compose_affine(delta=None, direction=None, theta=None, seq=None, degrees=True):
    """Compose an affine matrix in homogeneous coordinates from a translation vector
        and Euler angles.
    Args:
        delta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the magnitude of translation along a single axis, or a list, tuple, array
            or tensor which specifies the translation components for all three directions. Default is None.
        direction (str, optional): Specifies the direction of translation if delta is an int or float. Must be
            either 'x', 'y' or 'z', corresponding to one of three array axes. If off-axis translation is desired,
            please specify delta as a length-3 object (e.g. list, array) of translation components. Default is None.
        theta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the angle of rotation about a single axis, or a list, tuple, array or tensor which specifies
            a set of three Euler angles for rotation. Default is None.
        seq (str): Must be specified if theta is provided. Specifies sequencce of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call. This description is repeated from
            the documentation for scipy.spatial.transform.Rotation.from_euler. Default is None.
        degrees (bool, optional): If True, then the given angles are assumed to be in degrees.
            This description is repeated from the documentation for
            scipy.spatial.transform.Rotation.from_euler. Default is True.
    Returns:
        A (torch.Tensor): Affine matrix in homogeneous coordinates.
    """
    if delta is not None:
        assert isinstance(delta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'delta should be an int, float, list, tuple, array, or tensor.'
        if isinstance(delta, (int, float)):
            assert direction is not None, 'If delta is int or float, direction must be specified.'
            assert isinstance(direction, str), 'Translation direction must be "x", "y", or "z".'
            assert direction in 'xyz', 'Translation direction must be "x", "y", or "z".'
        else:
            if isinstance(delta, (np.ndarray, torch.Tensor)):
                assert len(delta.shape) == 1, 'If delta is an array or tensor it must have a single dimension.'
            assert len(delta) == 3, 'If delta is a list, tuple, array or tensor it must have length 3.'
    if theta is not None:
        assert isinstance(theta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'theta should be an int, float, list, tuple, array, or tensor.'
        assert seq is not None, 'If theta is int or float, seq must be specified.'
        assert isinstance(seq, str), 'seq argument must be a string.'
        if isinstance(theta, (int, float)):
            assert len(seq) == 1, 'If theta is int or float, seq must be "x", "y", or "z".'
            assert seq in 'xyz', 'If theta is int or float, seq must be "x", "y", or "z".'
        else:
            assert len(seq) == 3, \
                'If theta is list, tuple, array or tensor, seq must be a ' \
                'length-3 string containing letters "x", "y", and "z".'
            assert all([letter in 'xyz' for letter in seq]), 'All letters in seq must be "x", "y", or "z".'
            if isinstance(theta, (np.ndarray, torch.Tensor)):
                assert len(theta.shape) == 1, 'If theta is an array or tensor it must have a single dimension.'
            assert len(theta) == 3, 'If theta is a list, tuple, array or tensor it must have length 3.'
    assert isinstance(degrees, bool), 'Degrees must be boolean.'
    A = torch.eye(4)
    if delta is not None:
        if isinstance(delta, (int, float)):
            if direction == 'x':
                A[0, 3] = delta
            elif direction == 'y':
                A[1, 3] = delta
            elif direction == 'z':
                A[2, 3] = delta
        else:
            if isinstance(delta, (list, tuple, np.ndarray)):
                delta = torch.Tensor(delta)
            A[:3, 3] = delta
    if theta is not None:
        r = R.from_euler(seq, theta, degrees=degrees).as_matrix()
        r = torch.Tensor(r)
        A[:3, :3] = r
    return A


def resample_from_affine(img, affine, mode='bilinear', padding_mode='zeros', align_corners=False,
                         numpy_out=False):
    """Apply an affine transform to an image using resampling.
    Args:
        img (ndarray or torch.Tensor): Image to be transformed and resampled. Must be 3D or 4D
            with a channel dimension i.e. (C, D, H, W).
        affine (ndarray or torch.Tensor): Affine matrix to be applied in homogeneous coordinates.
            Must have shape (4, 4).
        mode (str, optional): Interpolation mode to calculate output values 'bilinear' | 'nearest'.
            Note that for 3D image input the interpolation mode used internally by torch is
            actually trilinear. Default is 'bilinear'
        padding_mode (str, optional): Padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
            Default is 'zeros'. See torch documentation for more details.
        align_corners (bool, optional): See torch documentation for details. Default is False.
        numpy_out (bool, optional): If True, return output as a numpy array. Default is False.
    Returns:
        out (ndarray or torch.Tensor): The resampled image. Shape is the same as the input.
    """
    assert isinstance(img, (np.ndarray, torch.Tensor)), 'Image must be ndarray or tensor.'
    assert isinstance(affine, (np.ndarray, torch.Tensor)), 'Affine matrix must be ndarray or tensor.'
    assert len(img.shape) in (3,4), 'Image must be a 3D or 4D array or tensor.'
    assert affine.shape == (4,4), 'Affine matrix must have shape (4,4).'
    img_dim = len(img.shape)   # input image dim
    if isinstance(img, np.ndarray):
        img = torch.Tensor(img)
    if len(img.shape) == 3:
        img = img.unsqueeze(dim=0)   # add channel dimension
    img = img.unsqueeze(dim=0)   # add batch dimension (always 1 for this function)
    if isinstance(affine, np.ndarray):
        affine = torch.Tensor(affine)
    affine = affine[:3, :]   # must supply shape (N,3,4) to affine_grid()
    affine = affine.unsqueeze(dim=0)   # must supply shape (N,3,4) to affine_grid()
    out_size = img.shape
    grid = torchF.affine_grid(affine, out_size, align_corners)
    out = torchF.grid_sample(img, grid, mode, padding_mode, align_corners)
    out = out.squeeze()
    if len(out.shape) == 3 and img_dim == 4:
        out = out.unsqueeze(dim=0)   # in case channel dimension is 1
    if numpy_out:
        return out.numpy()
    else:
        return out


def resample_from_affine_params(volume, delta=None, direction=None, pixels=True, theta=None, seq=None, degrees=True,
                                mode='bilinear', padding_mode='zeros', align_corners=False):
    """"Apply an affine transform to an image using resampling. The affine transform is specified
        by translation and rotation parameters.
    Args:
        volume (torch.Tensor): Volume to be transformed and resampled. Must be 4D
            with a channel dimension i.e. (C, D, H, W).
        delta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the magnitude of translation along a single axis, or a list, tuple, array
            or tensor which specifies the translation components for all three directions. Default is None.
        direction (str, optional): Specifies the direction of translation if delta is an int or float. Must be
            either 'x', 'y' or 'z', corresponding to one of three array axes. If off-axis translation is desired,
            please specify delta as a length-3 item of translation components. Default is None.
        pixels (bool, optional): If True, the magnitude of translation is specified in pixels, as opposed to
            units of half the input tensor (see pytorch grid_sample for details). Default is True.
        theta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the angle of rotation about a single axis, or a list, tuple, array or tensor which specifies
            a set of three Euler angles for rotation. Default is None.
        seq (str): Must be specified if theta is provided. Specifies sequencce of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call. This description is repeated from
            the documentation for scipy.spatial.transform.Rotation.from_euler. Default is None.
        degrees (bool, optional): If True, then the given angles are assumed to be in degrees.
            This description is repeated from the documentation for
            scipy.spatial.transform.Rotation.from_euler. Default is True.
        mode (str, optional): Interpolation mode to calculate output values 'bilinear' | 'nearest'.
            Note that for 3D image input the interpolation mode used internally by torch is
            actually trilinear. Default is 'bilinear'
        padding_mode (str, optional): Padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
            Default is 'zeros'. See torch documentation for more details.
        align_corners (bool, optional): See torch documentation for details. Default is False.
    Returns:
        volume (torch.Tensor): The resampled image. Shape is the same as the input.
    """
    assert _is_tensor_image_volume(volume), 'volume should be a 4D torch.Tensor with a channel dimension'
    if delta is not None:
        assert isinstance(delta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'delta should be an int, float, list, tuple, array, or tensor.'
        if isinstance(delta, (int, float)):
            assert direction is not None, 'If delta is int or float, direction must be specified.'
            assert isinstance(direction, str), 'Translation direction must be "x", "y", or "z".'
            assert direction in 'xyz', 'Translation direction must be "x", "y", or "z".'
        else:
            if isinstance(delta, (np.ndarray, torch.Tensor)):
                assert len(delta.shape) == 1, 'If delta is an array or tensor it must have a single dimension.'
            assert len(delta) == 3, 'If delta is a list, tuple, array or tensor it must have length 3.'
    assert isinstance(pixels, bool), 'pixels must be either True or False'
    if theta is not None:
        assert isinstance(theta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'theta should be an int, float, list, tuple, array, or tensor.'
        assert seq is not None, 'If theta is int or float, seq must be specified.'
        assert isinstance(seq, str), 'seq argument must be a string.'
        if isinstance(theta, (int, float)):
            assert len(seq) == 1, 'If theta is int or float, seq must be "x", "y", or "z".'
            assert seq in 'xyz', 'If theta is int or float, seq must be "x", "y", or "z".'
        else:
            assert len(seq) == 3, \
                'If theta is list, tuple, array or tensor, seq must ' \
                'be a length-3 string containing letters "x", "y", and "z".'
            assert all([letter in 'xyz' for letter in seq]), 'All letters in seq must be "x", "y", or "z".'
            if isinstance(theta, (np.ndarray, torch.Tensor)):
                assert len(theta.shape) == 1, 'If theta is an array or tensor it must have a single dimension.'
            assert len(theta) == 3, 'If theta is a list, tuple, array or tensor it must have length 3.'
    assert isinstance(degrees, bool), 'Degrees must be a boolean.'
    assert isinstance(mode, str), 'Mode must be "bilinear" or "nearest"'
    assert mode in ('bilinear', 'nearest'), 'Mode must be "bilinear" or "nearest"'
    assert isinstance(padding_mode, str), 'Padding mode must be either "zeros", "border" or "reflection"'
    assert padding_mode in ('zeros', 'border', 'reflection'), 'Padding mode must ' \
                                                              'be either "zeros", "border" or "reflection"'
    assert isinstance(align_corners, bool), 'align_corners must be True or False'
    if delta is not None and pixels is True:
        if isinstance(delta, (int, float)):
            if direction == 'x':
                direction_len = volume.shape[1]
            elif direction == 'y':
                direction_len = volume.shape[2]
            else:
                direction_len = volume.shape[3]
            delta = 2*delta/direction_len
        else:
            if isinstance(delta, (list, tuple, np.ndarray)):
                delta = torch.Tensor(delta)
            vol_shape = torch.Tensor(tuple(volume.shape[1:]))
            delta = 2*delta/vol_shape
    affine = compose_affine(delta, direction, theta, seq, degrees)
    volume = resample_from_affine(volume, affine, mode, padding_mode, align_corners)
    return volume


def apply_motion_from_affine_params(volume, time, delta=None, direction=None, pixels=True, theta=None, seq=None,
                                    degrees=True, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    Args:
        volume (torch.Tensor): Volume to be transformed and resampled. Must be 4D
            with a channel dimension i.e. (C, D, H, W).
        time (float): Time at which the motion occurs during scanning. Should be between [0.5, 1), where 0
            represents the beginning of the scan and 1 represents the end. Time >= 0.5 assures that the
            most prominent object in the image is in the original position of the image so that ground truth
            annotations don't need to be adjusted.
        delta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the magnitude of translation along a single axis, or a list, tuple, array
            or tensor which specifies the translation components for all three directions. Default is None.
        direction (str, optional): Specifies the direction of translation if delta is an int or float. Must be
            either 'x', 'y' or 'z', corresponding to one of three array axes. If off-axis translation is desired,
            please specify delta as a length-3 item of translation components. Default is None.
        pixels (bool, optional): If True, the magnitude of translation is specified in pixels, as opposed to
            units of half the input tensor (see pytorch grid_sample for details). Default is True.
        theta (int, float, list, tuple, np.ndarray, torch.Tensor, optional): Can either be a number (int or float)
            which specifies the angle of rotation about a single axis, or a list, tuple, array or tensor which specifies
            a set of three Euler angles for rotation. Default is None.
        seq (str): Must be specified if theta is provided. Specifies sequencce of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or {'x', 'y', 'z'} for extrinsic rotations.
            Extrinsic and intrinsic rotations cannot be mixed in one function call. This description is repeated from
            the documentation for scipy.spatial.transform.Rotation.from_euler. Default is None.
        degrees (bool, optional): If True, then the given angles are assumed to be in degrees.
            This description is repeated from the documentation for
            scipy.spatial.transform.Rotation.from_euler. Default is True.
        mode (str, optional): Interpolation mode to calculate output values 'bilinear' | 'nearest'.
            Note that for 3D image input the interpolation mode used internally by torch is
            actually trilinear. Default is 'bilinear'
        padding_mode (str, optional): Padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
            Default is 'zeros'. See torch documentation for more details.
        align_corners (bool, optional): See torch documentation for details. Default is False.
    Returns:
        volume (torch.Tensor): Motion-artifacted image. Shape is the same as the input.
    """
    if delta is not None:
        assert isinstance(delta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'delta should be an int, float, list, tuple, array, or tensor.'
        if isinstance(delta, (int, float)):
            assert direction is not None, 'If delta is int or float, direction must be specified.'
            assert isinstance(direction, str), 'Translation direction must be "x", "y", or "z".'
            assert direction in 'xyz', 'Translation direction must be "x", "y", or "z".'
        else:
            if isinstance(delta, (np.ndarray, torch.Tensor)):
                assert len(delta.shape) == 1, 'If delta is an array or tensor it must have a single dimension.'
            assert len(delta) == 3, 'If delta is a list, tuple, array or tensor it must have length 3.'
    assert isinstance(pixels, bool), 'pixels must be either True or False'
    if theta is not None:
        assert isinstance(theta, (int, float, list, tuple, np.ndarray, torch.Tensor)), \
            'theta should be an int, float, list, tuple, array, or tensor.'
        assert seq is not None, 'If theta is int or float, seq must be specified.'
        assert isinstance(seq, str), 'seq argument must be a string.'
        if isinstance(theta, (int, float)):
            assert len(seq) == 1, 'If theta is int or float, seq must be "x", "y", or "z".'
            assert seq in 'xyz', 'If theta is int or float, seq must be "x", "y", or "z".'
        else:
            assert len(seq) == 3, \
                'If theta is list, tuple, array or tensor, seq must ' \
                'be a length-3 string containing letters "x", "y", and "z".'
            assert all([letter in 'xyz' for letter in seq]), 'All letters in seq must be "x", "y", or "z".'
            if isinstance(theta, (np.ndarray, torch.Tensor)):
                assert len(theta.shape) == 1, 'If theta is an array or tensor it must have a single dimension.'
            assert len(theta) == 3, 'If theta is a list, tuple, array or tensor it must have length 3.'
    assert isinstance(degrees, bool), 'Degrees must be a boolean.'
    assert isinstance(mode, str), 'Mode must be "bilinear" or "nearest"'
    assert mode in ('bilinear', 'nearest'), 'Mode must be "bilinear" or "nearest"'
    assert isinstance(padding_mode, str), 'Padding mode must be either "zeros", "border" or "reflection"'
    assert padding_mode in ('zeros', 'border', 'reflection'), 'Padding mode must be ' \
                                                              'either "zeros", "border" or "reflection"'
    assert isinstance(align_corners, bool), 'align_corners must be True or False'
    n_vox = volume[0].numel()   # get number of voxels in first channel
    time = round(time*n_vox)   # get voxel where movement occurs (k-space filling)
    time = np.array([time])   # put in array to be fed into mask constructor function
    masks = construct_kspace_masks(volume, time)
    fft = torch.rfft(volume, signal_ndim=3, onesided=False)
    fft = ks_motion_fftshift(fft, axes=(1, 2, 3))
    fft = complex_from_split(fft)
    k = masks[0]*fft   # initialize composite k-space
    volume = resample_from_affine_params(volume, delta, direction, pixels, theta, seq, degrees,
                                                   mode, padding_mode, align_corners)
    fft = torch.rfft(volume, signal_ndim=3, onesided=False)
    fft = ks_motion_fftshift(fft, axes=(1, 2, 3))
    fft = complex_from_split(fft)
    k += masks[1]*fft
    k = split_from_complex(k)
    k = ks_motion_ifftshift(k, axes=(1, 2, 3))
    volume = torch.irfft(k, signal_ndim=3, onesided=False)
    return volume
