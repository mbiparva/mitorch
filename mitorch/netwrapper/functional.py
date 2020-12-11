#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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


# modified from Kornia multi-class to
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
def focal_loss_kornia(
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

    # compute sigmoid
    input_soft: torch.Tensor = F.sigmoid(input) + eps
    input_soft = input_soft.squeeze(dim=1)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    floss = -alpha * weight * torch.log(input_soft)

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


# modified from torchvision 0.8.2
# https://github.com/pytorch/vision/blob/v0.8.2/torchvision/ops/focal_loss.py
def focal_loss_torchvision(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Arguments:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


def rvd(
        input: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int) -> torch.tensor:
    """
    This function implements Relative Volume Distance metric. Possible values are [-1, +inf].
    It assumes the input and target have 0 and 1 values reminiscent of binary tensors.
    Args:
        ignore_index: the value to ignore applying the metric on
        input: input tensor predicted by the model
        target: target tensor provided as the ground truth

    Returns: metric value

    """
    assert input.size() == target.size(), 'input size and target size do not match'

    apply_ignore_index(input, target, ignore_index, fill_value=0)

    input_volume = input.sum()
    target_volume = target.sum()

    if not target_volume:
        raise RuntimeError('The target tensor does not contain any binary object.')

    return (input_volume - target_volume) / float(target_volume)


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
