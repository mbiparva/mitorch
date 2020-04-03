#!/usr/bin/env python3

from .build import DATASET_REGISTRY
from torchvision.datasets import MNIST


@DATASET_REGISTRY.register()
class MNIST(MNIST):
    def __init__(self, cfg, split, transformations):
        super().__init__(
            cfg.PROJECT.DATASET_DIR,
            train=True,
            transform=transformations,
            target_transform=None,
            download=True
        )
