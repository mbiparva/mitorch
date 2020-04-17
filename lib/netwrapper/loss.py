from torch.nn.modules.loss import _WeightedLoss
from netwrapper.functional import dice_coeff


# noinspection PyArgumentList
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
