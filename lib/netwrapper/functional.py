import torch


# noinspection PyPep8Naming
def dice_coeff(input, target, weight=None, ignore_index=-100, reduction='mean', epsilon=1e-6, standard=True):
    assert (input.is_contiguous() and target.is_contiguous())
    # TODO check to see if contiguous() is needed anywhere in the function
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    assert input.dtype() == target.dtype(), 'dtype does not match'
    N, C = input.shape[:2]

    if ignore_index > 0:
        ignore_mask = target.eq(ignore_index)
        if ignore_mask.any():
            input.masked_fill(ignore_mask, 0)
            target.masked_fill(ignore_mask, 0)
    assert (input.is_contiguous() and target.is_contiguous())

    intersect = input * target
    if standard:
        union = input + target
    else:
        union = input.pow(2) + target.pow(2)
    assert (intersect.is_contiguous() and union.is_contiguous())

    intersect = intersect.contiguous().view(N, C, -1).sum(dim=2)
    union = union.contiguous().view(N, C, -1).sum(dim=2).clamp(min=epsilon)
    assert (input.is_contiguous() and target.is_contiguous())

    if weight is not None:
        intersect = weight * intersect
    assert (input.is_contiguous() and target.is_contiguous())

    dice_coefficient = 2 * intersect / union

    dice_coefficient = dice_coefficient.mean(dim=1)  # class dimension

    return {
        'mean': dice_coefficient.mean(),
        'sum': dice_coefficient.sum(),
        'none': dice_coefficient
    }[reduction]
