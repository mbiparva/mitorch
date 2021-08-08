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


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


batch = 10
channel = 3
depth = 30
width = 512
height = 512

rand_tensor = torch.rand((batch, channel, depth, height, width))

print(rand_tensor.size())

net = AuxillaryHead(mode='mean', _map=True)

output_tensor = net(rand_tensor)

print(output_tensor.size())

loss_function = CrossEntropyLoss()

loss = loss_function(input=output_tensor, target=output_tensor)

print(loss)
