#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import numpy as np
import nibabel as nib
import data.transforms_mitorch as tf
from torch.utils.data import DataLoader
import torchvision.transforms as torch_tf
from data.data_container import ds_worker_init_fn
from data.VolSet import collate_fn
from data.TestSetExt import TestSet
from models.build import build_model
import utils.checkpoint as checkops
from data.build import build_dataset
from utils.metrics import dice_coefficient_metric, jaccard_index_metric, hausdorff_distance_metric


def binarize_pred(p, binarize_threshold):
    prediction_mask = p.ge(binarize_threshold)
    p = p.masked_fill(prediction_mask, 1)
    p = p.masked_fill(~prediction_mask, 0)

    return p


def save_pred(pred, output_dir):
    pred = pred.cpu().numpy()
    pred = nib.Nifti1Image(pred, np.eye(4))
    nib.save(pred, output_dir)


# TODO do it in functional or OOP later
def eval_pred(p, a, meters, cfg):
    meters['dice_coeff'] = dice_coefficient_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)
    meters['jaccard_ind'] = jaccard_index_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)
    meters['hausdorff_dist'] = hausdorff_distance_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)


def test(cfg):
    # (0) initial setup
    cfg.TRAIN.ENABLE = cfg.VALID.ENABLE = False
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

    # (2) define data pipeline
    evaluate_pred = True
    transformations = torch_tf.Compose([
        tf.ToTensorImageVolume(),
        tf.RandomOrientationTo('RPI'),
        tf.RandomResampleTomm(target_spacing=(1, 1, 1)),

        tf.ResizeImageVolume(cfg.DATA.MAX_SIDE_SIZE, min_side=cfg.DATA.MIN_SIDE),
        tf.PadToSizeVolume(cfg.DATA.MAX_SIDE_SIZE, padding_mode=cfg.DATA.PADDING_MODE),

        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
        tf.NormalizeMeanStdVolume(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, inplace=True),
    ])
    # Define any test dataset with annotation as known dataset otherwise call TestSet
    if len(cfg.TEST.DATA_PATH):
        evaluate_pred = False
        cfg.TEST.IN_MOD = [  # TODO if needed, we can add this to the input arguments
            ('t1', 'T1.nii.gz'),
            ('fl', 'FLAIR.nii.gz'),
            # ('annot', 'wmh.nii.gz'),
        ]
        test_set = TestSet(cfg, 'test', transformations)
    else:
        test_set = build_dataset(cfg.TEST.DATASET, cfg, 'test', transformations)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False,
                             num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                             pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                             worker_init_fn=ds_worker_init_fn,
                             collate_fn=collate_fn,
                             )

    # (3) create network and load snapshots
    net = build_model(cfg, 0)
    checkops.load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, net, data_parallel=False)

    # (4) loop over samples
    meters_test_set = list()
    for cnt, (image, annot, meta) in enumerate(test_loader):
        print(cnt, len(test_loader))
        meters = dict()

        image = image.to(device, non_blocking=True)

        # (A) Get prediction
        pred = net(image)

        # (B) Threshold prediction
        pred = binarize_pred(pred, binarize_threshold=cfg.TEST.BINARIZE_THRESHOLD)

        # (C) Save prediction
        save_pred(pred, cfg.OUTPUT_DIR)

        # (D) Evaluate prediction
        if evaluate_pred:
            eval_pred(pred, annot, meters, cfg)
            meters_test_set.append(meters)

    print('*** Done saving segmentation prediction for the test data.'
          '*** Results are saved at:')
    print(cfg.OUTPUT_DIR)

    if evaluate_pred:
        print('\nEvaluation results on the test set is:')
        for k in meters_test_set[0].keys():
            print(k, np.array([i[k] for i in meters_test_set]).mean())
