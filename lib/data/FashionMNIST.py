#!/usr/bin/env python3

from .build import DATASET_REGISTRY
from torchvision.datasets import FashionMNIST


@DATASET_REGISTRY.register()
class FashionMNIST(FashionMNIST):
    def __init__(self, cfg, split, transformations):
        super().__init__(
            cfg.PROJECT.DATASET_DIR,
            train=True,
            transform=transformations,
            target_transform=None,
            download=True
        )
