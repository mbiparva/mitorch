#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import data.transforms_mitorch as tf
import torchvision.transforms as torch_tf
from models.build import build_model
import utils.checkpoint as checkops
import torch
import numpy as np


# I choose to use flat implementation
# TODO do it in functional or OOP later
def test(cfg):
    # (0) initial setup
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cuda_device_id = cfg.GPU_ID
    if cfg.NUM_GPUS > 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cuda_device_id))
        print('cuda available')
        print('device count is', torch.cuda.device_count())
        print(device, 'will be used ...')
    else:
        device = torch.device('cpu')

    # (1) setup data root
    assert cfg.TEST.DATA_PATH and len(cfg.TEST.DATA_PATH), 'TEST.DATA_PATH not set'
    assert cfg.TEST.CHECKPOINT_FILE_PATH and len(cfg.TEST.CHECKPOINT_FILE_PATH), 'TEST.CHECKPOINT_FILE_PATH not set'
    print('you chose {} mode'.format(('single', 'batch')[cfg.TEST.BATCH_MODE]))
    test_data_path_list = [cfg.TEST.DATA_PATH] if not cfg.TEST.BATCH_MODE else [
        os.path.join(cfg.TEST.DATA_PATH, i) for i in os.listdir(cfg.TEST.DATA_PATH)
    ]

    # (2) define data pipeline
    transformations = torch_tf.Compose([
        tf.ToTensorImageVolume(),
        tf.RandomOrientationTo('RPI'),
        tf.RandomResampleTomm(target_spacing=(1, 1, 1)),

        tf.ResizeImageVolume(cfg.DATA.MAX_SIDE_SIZE, min_side=cfg.DATA.MIN_SIDE),
        tf.PadToSizeVolume(cfg.DATA.MAX_SIDE_SIZE, padding_mode=cfg.DATA.PADDING_MODE),

        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
        tf.NormalizeMeanStdVolume(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, inplace=True),
    ])

    # (3) create network and load snapshots
    net = build_model(cfg, 0)
    checkops.load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, net, data_parallel=False)

    # (4) loop over samples
    for i, d in enumerate(test_data_path_list):
        print(i, len(test_data_path_list))

        # (A) load data

        # (B) Transform it

        # (C) Get prediction

        # (D) Save the segmentation result

