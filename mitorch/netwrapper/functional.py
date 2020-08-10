#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import scipy.ndimage
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn as nn
import torch.nn.functional as F


def apply_ignore_index(input, target, ignore_index, fill_value=0):
    if ignore_index > 0:
        ignore_mask = target.eq(ignore_index)
        if ignore_mask.any():
            input.masked_fill(ignore_mask, fill_value)  # TODO find a better way, exclude rather than replace
            target.masked_fill(ignore_mask, fill_value)


# noinspection PyPep8Naming
def dice_coeff(input, target, weight=None, ignore_index=-100, reduction='mean', epsilon=1e-6, standard=True):
    assert (input.is_contiguous() and target.is_contiguous())
    # TODO check to see if contiguous() is needed anywhere in the function
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    assert input.dtype == target.dtype, 'dtype does not match'

    apply_ignore_index(input, target, ignore_index, fill_value=0)

    assert (input.is_contiguous() and target.is_contiguous())

    intersect = input * target
    if standard:
        union = input + target
    else:
        union = input.pow(2) + target.pow(2)
    assert (intersect.is_contiguous() and union.is_contiguous())

    N, C = input.shape[:2]
    intersect = intersect.contiguous().view(N, C, -1).sum(dim=2)
    union = union.contiguous().view(N, C, -1).sum(dim=2).clamp(min=epsilon)
    assert (input.is_contiguous() and target.is_contiguous())

    if weight is not None:
        intersect = weight * intersect
    assert (input.is_contiguous() and target.is_contiguous())

    dice_coefficient = 2 * intersect / union

    dice_coefficient = dice_coefficient.mean(dim=1)  # class dimension

    return 1 - {
        'mean': dice_coefficient.mean(),
        'sum': dice_coefficient.sum(),
        'none': dice_coefficient
    }[reduction]


# noinspection PyPep8Naming
def jaccard_index(input, target, ignore_index=-100, reduction='mean', epsilon=1e-6):
    assert (input.is_contiguous() and target.is_contiguous())
    # TODO check to see if contiguous() is needed anywhere in the function
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    assert input.dtype == target.dtype, 'dtype does not match'

    apply_ignore_index(input, target, ignore_index, fill_value=0)

    assert (input.is_contiguous() and target.is_contiguous())

    intersect = input * target
    union = input + target
    assert (intersect.is_contiguous() and union.is_contiguous())

    N, C = input.shape[:2]
    intersect = intersect.contiguous().view(N, C, -1).sum(dim=2)
    union = union.contiguous().view(N, C, -1).sum(dim=2).clamp(min=epsilon)
    assert (input.is_contiguous() and target.is_contiguous())

    # TODO check dice_coeff for a class-weighted step

    jaccard_index_metric = intersect / (union - intersect).clamp(min=epsilon)

    jaccard_index_metric = jaccard_index_metric.mean(dim=1)  # class dimension

    return 1 - {
        'mean': jaccard_index_metric.mean(),
        'sum': jaccard_index_metric.sum(),
        'none': jaccard_index_metric
    }[reduction]


# noinspection PyBroadException
def _hausdorff_distance_func(e_1, e_2, maximum=True, percentile=95):
    """This is based on the scipy.ndimage.morphology package. Check scikit-video for the reference implementation.
    https://github.com/scikit-video/scikit-video/blob/master/skvideo/motion/gme.py
    """
    LARGE_NUMBER = 1000
    assert isinstance(e_1, np.ndarray) and isinstance(e_2, np.ndarray), 'expect np.ndarray ' \
                                                                        'but got {}'.format(type(e_1), type(e_2))
    # binary structure
    diamond = scipy.ndimage.generate_binary_structure(3, 1)

    # extract only 1-pixel border line of objects
    e_1_per = e_1 ^ scipy.ndimage.morphology.binary_erosion(e_1, structure=diamond)
    e_2_per = e_2 ^ scipy.ndimage.morphology.binary_erosion(e_2, structure=diamond)

    # Max of euclidean distance transform
    one_from_two = scipy.ndimage.morphology.distance_transform_edt(~e_2_per)[e_1_per]
    two_from_one = scipy.ndimage.morphology.distance_transform_edt(~e_1_per)[e_2_per]

    if maximum:
        one_from_two = one_from_two.max()
        two_from_one = two_from_one.max()
    else:
        one_from_two = np.percentile(one_from_two, percentile) if len(one_from_two) else LARGE_NUMBER
        # it seems this makes issue and since gt is empty, I set 0
        two_from_one = np.percentile(two_from_one, percentile) if len(two_from_one) else 0
        # try:  # IndexError is raised at BOAX with "cannot do a non-empty take from an empty axes"
        #     one_from_two = np.percentile(one_from_two, percentile)
        # except Exception as e:
        #     print('CAUGHT:', e)
        #     print('LARGE_NUMBER assigned instead')
        #     one_from_two = LARGE_NUMBER
        # try:  # IndexError is raised at BOAX with "cannot do a non-empty take from an empty axes"
        #     two_from_one = np.percentile(two_from_one, percentile)
        # except Exception as e:
        #     if not e_2_per.any():
        #         print('annot is all zero hence h95 metric is empty for two_from_one')
        #         two_from_one = 0
        #     else:
        #         print('CAUGHT:', e)
        #         print('LARGE_NUMBER assigned instead')
        #         two_from_one = LARGE_NUMBER
    # TODO we can even use mean

    return np.max((one_from_two, two_from_one))


def _hausdorff_distance_prep(input, target):
    # In and out of Numpy/Scipy Scope
    N, C = input.shape[:2]
    input_np, target_np = input.numpy(), target.numpy()
    hausdorff_distance_output = np.zeros((N, C))
    for i in range(N):
        for j in range(C):
            hausdorff_distance_output[i, j] = _hausdorff_distance_func(input_np[i, j], target_np[i, j],
                                                                       maximum=False, percentile=90)
    hausdorff_distance_output = torch.from_numpy(hausdorff_distance_output)

    return hausdorff_distance_output


# noinspection PyPep8Naming
def hausdorff_distance(input, target, ignore_index=-100, reduction='mean'):
    assert (input.is_contiguous() and target.is_contiguous())
    # TODO check to see if contiguous() is needed anywhere in the function
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    assert input.dtype == target.dtype, 'dtype does not match'

    apply_ignore_index(input, target, ignore_index, fill_value=0)

    assert (input.is_contiguous() and target.is_contiguous())

    input, target = input.bool(), target.bool()

    hausdorff_distance_output = _hausdorff_distance_prep(input, target)

    hausdorff_distance_output = hausdorff_distance_output.float()

    return - {
        'mean': hausdorff_distance_output.mean(),
        'sum': hausdorff_distance_output.sum(),
        'none': hausdorff_distance_output
    }[reduction]


# modified from Kornia
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    # input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
    # input = input + eps  # input is already passed to Sigmoid. It is binary classification
    input = input.squeeze(dim=1)

    # create the labels one hot tensor
    # target_one_hot: torch.Tensor = one_hot(
    #     target, num_classes=input.shape[1],
    #     device=input.device, dtype=input.dtype)
    # target_one_hot = target.unsqueeze(dim=1)  # target is already one-hot since it is a binary classification problem

    # loss = target_one_hot * torch.log(input_soft)
    loss = nn.BCELoss(reduction='none')(input, target)
    # loss = F.binary_cross_entropy(input, target, reduction='none')

    # compute the actual focal loss
    weight = torch.pow(-input + 1., gamma)
    focal = alpha * weight

    floss = focal * loss

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        floss = torch.mean(floss)
    elif reduction == 'sum':
        floss = torch.sum(floss)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return floss


if __name__ == '__main__':
    from PIL import Image
    input_test = Image.open('/home/mbiparva/Downloads/input.jpg')
    input_test = np.asanyarray(input_test)
    input_test = torch.from_numpy(input_test).float()[:, :, 0]
    input_test = torch.stack([input_test for _ in range(10)])
    input_test[input_test < 100] = 1
    input_test[input_test >= 100] = 0
    input_test = input_test.bool()
    input_test = input_test.unsqueeze(dim=0).unsqueeze(dim=0)

    target_test = Image.open('/home/mbiparva/Downloads/target.jpg')
    target_test = np.asanyarray(target_test)
    target_test = torch.from_numpy(target_test).float()[:, :, 0]
    target_test = torch.stack([target_test for _ in range(10)])
    target_test[target_test < 100] = 1
    target_test[target_test >= 100] = 0
    target_test = target_test.bool()
    target_test = target_test.unsqueeze(dim=0).unsqueeze(dim=0)

    dc = 1 - dice_coeff(input_test.float(), target_test.float())

    jc = 1 - jaccard_index(input_test.float(), target_test.float())

    hd = -hausdorff_distance(input_test, target_test)

    print(
        '********\n\n'
        'jaccard_index: {jaccard_index}\n'
        'dice_coeff: {dice_coeff}\n'
        'hausdorff_distance: {hausdorff_distance}\n\n'
        '********'.format(
            dice_coeff=dc,
            jaccard_index=jc,
            hausdorff_distance=hd,
        )
    )
