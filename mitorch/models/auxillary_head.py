import torch.nn as nn

from utils.MONAI_networks.utils import MarkerLayer


class AuxillaryHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._create_net()

    def _create_net(self):
        pass

    def forward(self, x):
        output_list = list()
        for name, module in self.net.features._modules.items():
            x = module(x)
            if isinstance(module, MarkerLayer):
                output_list.append(x)
        return output_list
