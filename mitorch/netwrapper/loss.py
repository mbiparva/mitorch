#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

from torch.nn.modules.loss import _WeightedLoss, CrossEntropyLoss
from netwrapper.functional import dice_coeff
from .build import LOSS_REGISTRY


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
