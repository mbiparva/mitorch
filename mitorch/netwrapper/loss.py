#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from torch.nn.modules.loss import _Loss, _WeightedLoss, CrossEntropyLoss
from netwrapper.functional import dice_coeff, apply_ignore_index
from .build import LOSS_REGISTRY
import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch import nn
# TODO: Use cdist or pairwise_distances for other a comprehensive list of distances
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn.functional as F
from .functional import focal_loss


__all__ = [
    'CrossEntropyLoss',
    'DiceLoss',
    'WeightedHausdorffLoss',
    'LovaszLoss',
]


@LOSS_REGISTRY.register()
class CrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# noinspection PyArgumentList
@LOSS_REGISTRY.register()
class DiceLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, standard=True, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.standard = standard
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return dice_coeff(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            standard=True
        )


# https://arxiv.org/pdf/1806.07564.pdf
# https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/60debd891f1fb9a5fbab5fe0e14d428bbbb80993/object-locator/losses.py
# https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py
def cdist_custom_euclidean(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    # x = x.detach().cpu().numpy()
    # y = y.detach().cpu().numpy()

    distance = (x - y) ** 2
    distances = distance.sum(-1).sqrt()
    return distances


# noinspection PyArgumentList
@LOSS_REGISTRY.register()
class WeightedHausdorffLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']
    FILL_VALUE = 0

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-100, **kwargs):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.whl_num_depth_sheets = kwargs.get('whl_num_depth_sheets', 2)
        self.whl_seg_thr = kwargs.get('whl_seg_thr', 0.5)

    @staticmethod
    def create_coordinate_list(input_shape):
        assert len(input_shape) > 1
        shape_range = [np.arange(i) for i in input_shape]

        volume_coordinates = torch.from_numpy(cartesian(shape_range))

        return volume_coordinates

    @staticmethod
    def _boundary_extraction_approximation(x_in, y_in):
        import scipy.ndimage

        x = x_in.detach().cpu().numpy()
        y = y_in.detach().cpu().numpy()

        # binary structure
        # TODO control margin at the boundary with the diamond size
        diamond = scipy.ndimage.generate_binary_structure(3, 1)

        # extract only 1-pixel border line of objects
        x_boundary = x ^ scipy.ndimage.morphology.binary_erosion(x, structure=diamond)
        y_boundary = y ^ scipy.ndimage.morphology.binary_erosion(y, structure=diamond)

        x_boundary = torch.from_numpy(x_boundary).to(x_in.device)
        y_boundary = torch.from_numpy(y_boundary).to(x_in.device)

        return x_boundary, y_boundary

    @staticmethod
    def calculate_terms(input_b, target_b_nz, distance_matrix, max_distance):
        input_b_flat = input_b.flatten()
        input_b_repeated = input_b.view(-1, 1).repeat(1, len(target_b_nz))
        estimate_num_points = input_b.sum()

        eps = 1e-6

        # Weighted Hausdorff Distance
        input_term_b = (1 / (estimate_num_points + eps)) * (input_b_flat * torch.min(distance_matrix, 1)[0]).sum()

        PAPER_IMP = False
        if not PAPER_IMP:
            alpha = 4
            target_term_b = torch.min((distance_matrix + eps) / (input_b_repeated ** alpha + eps / max_distance), 0)[0]
            target_term_b = torch.clamp(target_term_b, 0, max_distance)
            target_term_b = torch.mean(target_term_b, 0)
        else:
            # https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py
            def generaliz_mean(tensor, dim, p=-9, keepdim=False):
                assert p < 0
                return torch.mean((tensor + eps) ** p, dim, keepdim=keepdim) ** (1. / p)

            POWER = -4  # what paper use
            weighted_d_matrix = (1 - input_b_repeated) * max_distance + input_b_repeated * distance_matrix
            target_term_b = generaliz_mean(weighted_d_matrix, p=POWER, dim=0, keepdim=False)
            target_term_b = torch.mean(target_term_b)

        return input_term_b, target_term_b

    def compute_terms(self, input_b, target_b, input_shape, input_device, max_distance):
        assert target_b.sum(), 'assertion raised since target_b == 0'

        CUSTOM_DISTANCE = True
        VERSION = (0, 1, 2)[2]  # 0 crashed out of memory with 64x3, 1 is a boundary approximation
        if VERSION == 0:
            NUM_ATTEMPTS = 10
            ATTEMPT_DIVISOR = 0.5
            WINDOW_SIZE = 0.3
            assert 0 < WINDOW_SIZE < 0.5
            LOWER_QUANTILE, UPPER_QUANTILE = 0.5 - WINDOW_SIZE, 0.5 + WINDOW_SIZE
            input_b_np = input_b.detach().cpu().numpy().flatten()
            target_b_nz, distance_matrix = None, None
            for i in range(NUM_ATTEMPTS):
                try:
                    # TODO can avoid create_coordinate_list and use input_b.ge(lower_bound)
                    # cuda throws out of memory error, use cpu, even that fails
                    # input_b_nz = volume_coordinates.float().to('cpu')
                    lower_bound = np.quantile(input_b_np, LOWER_QUANTILE)
                    upper_bound = np.quantile(input_b_np, UPPER_QUANTILE)
                    input_b_i = input_b.lt(lower_bound) | input_b.ge(upper_bound)
                    input_b_nz = input_b_i.nonzero().float().to('cpu')
                    target_b_nz = target_b.nonzero().float().to('cpu')

                    assert input_b_nz.numel()

                    if CUSTOM_DISTANCE:
                        distance_matrix = cdist_custom_euclidean(input_b_nz, target_b_nz).to('cuda')
                    else:
                        input_b_nz = input_b_nz.detach().cpu().numpy()
                        target_b_nz = target_b_nz.detach().cpu().numpy()
                        distance_matrix = pairwise_distances(input_b_nz, target_b_nz, metric='euclidean', n_jobs=8)
                    input_b = input_b[input_b_i]
                    break
                except RuntimeError as e:
                    print('this error is thrown at the distance matrix calculation', e)
                    print('try to tighten the lower and upper bounds by halt of the distances to the extremes')
                    LOWER_QUANTILE = LOWER_QUANTILE * ATTEMPT_DIVISOR
                    UPPER_QUANTILE = UPPER_QUANTILE + (1.0 - UPPER_QUANTILE) * ATTEMPT_DIVISOR
            assert distance_matrix is not None, 'attempts failed to fit distance_matrix into memory'
        elif VERSION == 1:
            # version one crashes too, there are two options:
            # (1) use for loop over depth
            # (2) use hausdorff after some epoch once it is stable, as an auxiliary term ---> chose this one ;)
            input_b_ge, target_b = input_b.ge(0.98), target_b.bool()
            input_b_boundary, target_b_boundary = self._boundary_extraction_approximation(input_b_ge, target_b)
            input_b_nz = input_b_boundary.nonzero().float()
            target_b_nz = target_b_boundary.nonzero().float()
            input_b = input_b[input_b_boundary]
            if CUSTOM_DISTANCE:
                input_b_nz = input_b_nz.to(input_device)
                target_b_nz = target_b_nz.to(input_device)
                distance_matrix = cdist_custom_euclidean(input_b_nz, target_b_nz)
            else:
                input_b_nz_np = input_b_nz.detach().cpu().numpy()
                target_b_nz_np = target_b_nz.detach().cpu().numpy()
                distance_matrix = pairwise_distances(input_b_nz_np, target_b_nz_np, metric='euclidean', n_jobs=12)
                distance_matrix = torch.from_numpy(distance_matrix).to(input_device)
        elif VERSION == 2:
            UNIFORM_IND_SAMPLING = (False, True)[1]

            # Use randomly weighted sampled depth sheets and measure loss at them.
            max_distance = (input_shape[1:].float() ** 2).sum().sqrt().item()
            input_term_b, target_term_b = list(), list()

            target_b_r_weights = target_b.sum(-1).sum(-1)
            assert target_b_r_weights.sum(), 'target_b_r_weights.sum() == 0'
            print(target_b_r_weights.sum())

            # TODO use some measure of overlap so the sheets with high error would have higher weights
            if UNIFORM_IND_SAMPLING:
                target_b_r_weights = target_b_r_weights.long().cpu().numpy()
                target_b_r_weights_nz_i = np.where(target_b_r_weights != 0)[0]
                depth_r_ind = np.random.choice(
                    target_b_r_weights_nz_i, min(self.whl_num_depth_sheets, len(target_b_r_weights_nz_i)), replace=False
                )
            else:
                depth_sampling_weights = target_b_r_weights / target_b_r_weights.sum()
                depth_r_ind = torch.multinomial(depth_sampling_weights, self.whl_num_depth_sheets, replacement=False)
            assert len(depth_r_ind)

            ALL_COORD = (False, True)[1]
            volume_coordinates = self.create_coordinate_list(input_shape[1:])
            input_b_ge, target_b = input_b.ge(self.whl_seg_thr), target_b.bool()

            for i in depth_r_ind:
                if ALL_COORD:
                    input_b_nz = volume_coordinates.float().to('cpu')
                    input_b_i = input_b[i]
                else:
                    input_b_ge_i = input_b_ge[i]
                    input_b_nz = input_b_ge_i.nonzero().float()
                    input_b_i = input_b[i, input_b_ge_i]
                if len(input_b_i) == 0:
                    print('**** one depth sheet got zero prediction ove the threshold ****')
                    continue
                target_b_i = target_b[i]
                target_b_nz = target_b_i.nonzero().float()
                # CUSTOM_DISTANCE
                input_b_nz = input_b_nz.to(input_device)
                target_b_nz = target_b_nz.to(input_device)
                distance_matrix = cdist_custom_euclidean(input_b_nz, target_b_nz)
                input_term_b_i, target_term_b_i = self.calculate_terms(input_b_i, target_b_nz,
                                                                       distance_matrix, max_distance)
                input_term_b.append(input_term_b_i)
                target_term_b.append(target_term_b_i)

            assert len(input_term_b) and len(target_term_b), 'got all depth sheets zero'
            input_term_b = torch.stack(input_term_b)
            target_term_b = torch.stack(target_term_b)

            input_term_b = input_term_b.mean()
            target_term_b = target_term_b.mean()
        else:
            raise NotImplementedError

        if not VERSION == 2:
            input_term_b, target_term_b = self.calculate_terms(input_b, target_b_nz, distance_matrix, max_distance)

        return input_term_b, target_term_b

    def forward(self, input, target):
        assert not target.requires_grad, 'target does not require grad'
        assert input.dim() == target.dim() == 5, 'Expect 4D tensor of BxDxHxW'
        assert input.shape == target.shape, 'Shapes must match'
        assert input.shape[1] == target.shape[1] == 1, 'it expects binary segmentation in one channel'
        assert 0 <= input.min().item() <= 1 and 0 <= input.max().item() <= 1, 'input values must be probabilities' \
                                                                              '{}, {}'.format(input.min().item(),
                                                                                              input.max().item())
        input, target = input.squeeze(1), target.squeeze(1)

        apply_ignore_index(input, target, self.ignore_index, fill_value=self.FILL_VALUE)

        batch_size = input.shape[0]
        input_shape = torch.tensor(input.shape[1:])  # DxHxW
        input_device = input.device

        # because of out-of-memory, I decided to use range finder for the indices on the either far sides of 0 or 1
        # unless I use a loop over the depth dimension, which is going to be slow, and sub-optimal
        # volume_coordinates = self.create_coordinate_list(input_shape)
        max_distance = (input_shape.float() ** 2).sum().sqrt().item()

        input_term, target_term = list(), list()
        for b in range(batch_size):
            input_b, target_b = input[b], target[b]

            try:
                input_term_b, target_term_b = self.compute_terms(
                    input_b,
                    target_b,
                    input_shape,
                    input_device,
                    max_distance
                )
            except Exception as e:
                print(f'exception {e} caught, address it')
                input_term_b = torch.tensor(0, device=input_device, dtype=torch.float, requires_grad=False)
                target_term_b = torch.tensor(max_distance, device=input_device, dtype=torch.float, requires_grad=False)

            input_term.append(input_term_b)
            target_term.append(target_term_b)

        input_term = torch.stack(input_term)
        target_term = torch.stack(target_term)

        loss = input_term + target_term

        return {
            'mean': loss.mean(),
            'sum': loss.sum(),
            'none': loss,
        }[self.reduction]


# modified from Kornia
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
class FocalLossKornia(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


@LOSS_REGISTRY.register()
class FocalLoss(FocalLossKornia):
    def __init__(self, ignore_index=None, **kwargs):
        self.ignore_index = ignore_index
        super().__init__(**kwargs)

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        B, C = input.shape[:2]
        input = input.reshape(B, C, -1)
        target = target.reshape(B, -1)

        return super().forward(input, target)


@LOSS_REGISTRY.register()
class LovaszLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, per_image=True, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.per_image = per_image
        self.ignore_index = ignore_index

    # All the lovasz prefixed functions are taken with minor modifications from the package below.
    # https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, D, H, W] Variable, logits at each pixel (between -infinity and +infinity)
          labels: [B, D, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        assert logits.ndim == labels.ndim

        if per_image:
            loss = self.lovasz_mean(
                self.lovasz_hinge_flat(
                    *self.lovasz_flatten_binary_scores(
                        log.unsqueeze(0),
                        lab.unsqueeze(0),
                        ignore,
                    )
                )
                for log, lab in zip(logits, labels)
            )
        else:
            loss = self.lovasz_hinge_flat(
                *self.lovasz_flatten_binary_scores(
                    logits,
                    labels,
                    ignore,
                )
            )
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -infinity and +infinity)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    @staticmethod
    def lovasz_flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    @staticmethod
    def lovasz_mean(loss, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        try:
            from itertools import ifilterfalse
        except ImportError:  # py3k
            from itertools import filterfalse as ifilterfalse

        def isnan(x):
            return x != x

        loss = iter(loss)
        if ignore_nan:
            loss = ifilterfalse(isnan, loss)
        try:
            n = 1
            acc = next(loss)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(loss, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(
            self,
            input: torch.tensor,
            target: torch.tensor) -> torch.tensor:

        assert (input.is_contiguous() and target.is_contiguous())
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        assert input.dtype == target.dtype, 'dtype does not match'

        return self.lovasz_hinge(
            input,
            target,
            per_image=self.per_image,
            ignore=self.ignore_index
        )
