import torch
import torch.nn as nn

# pip install einops==0.3.0
from einops import reduce

VALID_MODES = ['mean', 'max']


class AuxillaryHead(nn.Module):
    def __init__(self, mode: str, _map: bool = False):
        super().__init__()

        self.mode = mode
        self.map = _map

        self._create_net()

    def _create_net(self):
        pass

    def forward(self, x):
        reduced = self.__reduce(input=x)
        return self.__map(input=reduced) if self.map else reduced

    def __reduce(self, input: torch.Tensor) -> torch.Tensor:
        return reduce(input, 'b c d w h -> b d', self.mode)

    def __map(self, input: torch.Tensor) -> torch.Tensor:
        depth = input.size()[1]
        mapping = nn.Linear(depth, depth)
        return mapping(input)


# class SliceOrderingNet(nn.Module):
#     def __init__(self, net: nn.Module, mode: str):
#         super().__init__()

#         if mode not in VALID_MODES:
#             raise ValueError(f'Reduction mode `{mode}` is not valid.')
#         self.mode = mode

#         self.net = net()
#         self.auxillary_head = AuxillaryHead()

#         self._create_net()

#     def _create_net(self):
#         pass

#     # add a linear_mapping boolean argument
#     def forward(self, x):
#        head = x
