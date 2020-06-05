#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from torch.nn.modules.loss import _WeightedLoss, CrossEntropyLoss
from netwrapper.functional import dice_coeff
from .build import LOSS_REGISTRY

import math
import torch
from sklearn.utils.extmath import cartesian
import numpy as np
from torch import nn
# TODO: Use cdist or pairwise_distances for other a comprehensive list of distances
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances


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
def cdist_custom_euclidean(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    distance = (x - y) ** 2
    distances = distance.sum(-1).sqrt()
    return distances


# noinspection PyArgumentList
@LOSS_REGISTRY.register()
class AveragedHausdorffLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-100):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = cdist_custom_euclidean(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res

# noinspection PyArgumentList
class WeightedHausdorffDistance(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']
    def __init__(self,
                 weight=None, size_average=None, reduce=None, reduction='mean', ignore_index=-100):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    @staticmethod
    def create_coordinate_list(input_shape):
        input_depth, input_height, input_width = input_shape
        d_range, h_range, w_range = np.arange(input_depth), np.arange(input_height), np.arange(input_width)

        volume_coordinates = torch.from_numpy(
            cartesian([
                d_range,
                h_range,
                w_range
            ])
        )

        return volume_coordinates

    def forward(self, input, target):
        assert not target.requires_grad, 'target does not require grad'
        assert input.dim() == target.dim() == 4, 'Expect 4D tensor of BxDxHxW'
        assert input.shape == target.shape, 'Shapes must match'
        assert 0 <= input.min().item() <= 1 and 0 <= input.max().item() <= 1, 'input values must be probabilities'

        batch_size = input.shape[0]
        input_shape = torch.tensor(input.shape[1:])  # DxHxW
        input_device = input.device()

        volume_coordinates = self.create_coordinate_list(input_shape, input_device)
        max_distance = (input_shape ** 2).sum().sqrt().item()

        input_term = []
        target_term = []
        # TODO: tensorize the computation ;)
        for b in range(batch_size):
            input_b, target_b = input[b], target[b]

            # TODO can avoid create_coordinate_list and use input_b.ge(lower_bound)
            input_b_nz = volume_coordinates.float().to(input_device)
            target_b_nz = target_b.nonzero().float().to(input_device)

            distance_matrix = cdist_custom_euclidean(input_b_nz, target_b_nz)

            input_b_flat = input_b.flatten()
            input_b_repeated = input_b.view(-1, 1).repeat(1, target_b_nz.size(0))
            estimate_num_points = input_b.sum()

            eps = 1e-6
            alpha = 4

            # Weighted Hausdorff Distance
            input_term_b = (1 / (estimate_num_points + eps)) * (input_b_flat * torch.min(distance_matrix, 1)[0]).sum()

            PAPER_IMP = False
            if not PAPER_IMP:
                target_term_b = torch.min((distance_matrix + eps) / (input_b_repeated ** alpha + eps / max_distance), 0)[0]
                target_term_b = torch.clamp(target_term_b, 0, max_distance)
                target_term_b = torch.mean(target_term_b, 0)
            else:
                # https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py
                def generaliz_mean(tensor, dim, p=-9, keepdim=False):
                    assert p < 0
                    return torch.mean((tensor + eps) ** p, dim, keepdim=keepdim) ** (1. / p)
                POWER = -9
                weighted_d_matrix = (1 - input_b_repeated) * max_distance + input_b_repeated * distance_matrix
                target_term_b = generaliz_mean(weighted_d_matrix, p=POWER, dim=0, keepdim=False)
                target_term_b = torch.mean(target_term_b)

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
