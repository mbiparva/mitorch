#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

from .build import MODEL_REGISTRY, build_model
from .Unet3D import Unet3D
from .NestedUnet3D import NestedUnet3D
from .Unet3DCBAM import Unet3DCBAM
from .DAUNet3D import DAUnet3D
from .MBUnet3D import MBUnet3D
from .MUnet3D import MUnet3D
from .SEUnet3D import SEUnet3D
from .Vnet3D import Vnet3D


# TODO add https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/nets/densenet.py
# TODO add https://github.com/Project-MONAI/MONAI/blob/master/monai/networks/nets/dynunet.py
