#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
from .build import MODEL_REGISTRY
from .weight_init_helper import init_weights


@MODEL_REGISTRY.register()
class Unet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._create_net()

        self.init_weights()

    def _create_net(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def init_weights(self):
        init_weights(self, self.cfg.MODEL.FC_INIT_STD)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
