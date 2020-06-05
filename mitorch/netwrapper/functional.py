#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import torch
import scipy.ndimage
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


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


# https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/60debd891f1fb9a5fbab5fe0e14d428bbbb80993/object-locator/losses.py
# TODO check this in comparison to the custom in-use one
def averaged_hausdorff_distance(set_x, set_y, max_ahd=np.inf):
    set_x = np.array(set_x)
    set_y = np.array(set_y)

    assert len(set_x) and len(set_y)
    assert set_x.ndim == set_y.ndim == 2, 'ndim must be 2'
    assert set_x.shape[1] == set_y.shape[1], 'The points in both sets must have the same number of dimensions.'

    d2_matrix = pairwise_distances(set_x, set_y, metric='euclidean')

    dis_to_edges_x = np.min(d2_matrix, axis=0)
    dis_to_edges_y = np.min(d2_matrix, axis=1)

    dis_x = np.average(dis_to_edges_x)
    dis_y = np.average(dis_to_edges_y)

    return dis_x + dis_y


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
        try:  # IndexError is raised at BOAX with "cannot do a non-empty take from an empty axes"
            one_from_two = np.percentile(one_from_two, percentile)
            two_from_one = np.percentile(two_from_one, percentile)
        except Exception as e:
            print('CAUGHT:', e)
            print('LARGE_NUMBER assigned instead')
            one_from_two = LARGE_NUMBER
            two_from_one = LARGE_NUMBER

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
