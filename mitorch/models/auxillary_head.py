import torch
import torch.nn as nn

# pip install einops==0.3.0
from einops import reduce

VALID_MODES = ['mean', 'max']


class AuxillaryHead(nn.Module):
    def __init__(self, cfg, mode: str):
        super().__init__()

        if mode not in VALID_MODES:
            raise ValueError(f'Reduction mode `{mode}` is not valid.')
        self.mode = mode

        self._create_net()

    def _create_net(self):
        pass

    # add a linear_mapping boolean argument
    def forward(self, x):
        return self._reduce(input=x)
        # if linear_mapping:
        #     do nn.linear input=d output=2

    def __reduce(self, input: torch.Tensor) -> torch.Tensor:
        return reduce(input, 'b c d w h -> b d', self.mode)
