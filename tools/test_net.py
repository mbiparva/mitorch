#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
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
from config.defaults import init_cfg


def binarize_pred(p, binarize_threshold):
    prediction_mask = p.ge(binarize_threshold)
    p = p.masked_fill(prediction_mask, 1)
    p = p.masked_fill(~prediction_mask, 0)

    return p


def save_pred(pred, output_dir, basename, *in_mod):
    output_path = ''
    file_type = ('nii', 'img')[0]
    pred = pred.detach().cpu().numpy()[0, 0]  # batch size is 1, channel is 1 too.
    if file_type == 'nii':
        output_path = os.path.join(output_dir, '{}_mask_pred.nii.gz'.format(os.path.basename(basename)))
        pred = nib.Nifti1Image(pred, np.eye(4))
        if len(in_mod):
            img = in_mod[0].detach().cpu().numpy()[0, 0]  # batch size is 1, channel 0 is usually T1
            img_output_path = os.path.join(output_dir, '{}_T1.nii.gz'.format(os.path.basename(basename)))
            img = nib.Nifti1Image(img, np.eye(4))
            nib.save(img, img_output_path)
            img = in_mod[0].detach().cpu().numpy()[0, 1]  # batch size is 1, channel 0 is usually T1
            img_output_path = os.path.join(output_dir, '{}_FLAIR.nii.gz'.format(os.path.basename(basename)))
            img = nib.Nifti1Image(img, np.eye(4))
            nib.save(img, img_output_path)
    elif file_type == 'img':
        output_path = os.path.join(output_dir, '{}_mask_pred.img'.format(os.path.basename(basename)))
        pred = nib.AnalyzeImage(pred, np.eye(4))
    nib.save(pred, output_path)


# TODO do it in functional or OOP later
def eval_pred(p, a, meters, cfg):
    meters['dice_coeff'] = dice_coefficient_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)
    meters['jaccard_ind'] = jaccard_index_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)
    meters['hausdorff_dist'] = hausdorff_distance_metric(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)


def test(cfg):
    # (0) initial setup
    os.rmdir(cfg.OUTPUT_DIR)  # it is useless, we use hpo_output_dir instead
    cfg = init_cfg(cfg)

    cfg.TRAIN.ENABLE = cfg.VALID.ENABLE = False
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cuda_device_id = cfg.GPU_ID
    torch.cuda.set_device(cuda_device_id)
    if cfg.NUM_GPUS > 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cuda_device_id))
        print('cuda available')
        print('device count is', torch.cuda.device_count())
        print(device, 'will be used ...')
    else:
        device = torch.device('cpu')

    # (1) setup data root
    assert cfg.TEST.CHECKPOINT_FILE_PATH and len(cfg.TEST.CHECKPOINT_FILE_PATH), 'TEST.CHECKPOINT_FILE_PATH not set'

    # (2) define data pipeline
    eval_pred_flag = True
    save_pred_flag = True
    transformations = torch_tf.Compose([
        tf.ToTensorImageVolume(),
        tf.RandomOrientationTo('RPI'),
        tf.RandomResampleTomm(target_spacing=(1, 1, 1)),

        tf.ResizeImageVolume(cfg.DATA.MAX_SIDE_SIZE, min_side=cfg.DATA.MIN_SIDE),
        # tf.PadToSizeVolume(cfg.DATA.MAX_SIDE_SIZE, padding_mode=cfg.DATA.PADDING_MODE),

        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
        tf.NormalizeMeanStdVolume(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, inplace=True),
    ])
    # Define any test dataset with annotation as known dataset otherwise call TestSet
    if len(cfg.TEST.DATA_PATH):
        print('you chose {} mode'.format(('single', 'batch')[cfg.TEST.BATCH_MODE]))
        eval_pred_flag = False
        save_pred_flag = True

        os.rmdir(cfg.OUTPUT_DIR)  # it is useless, we use hpo_output_dir instead
        cfg.TEST.DATASET = 'TestSet{}'.format(('single', 'batch')[cfg.TEST.BATCH_MODE].upper())
        cfg = init_cfg(cfg)

        cfg.TEST.IN_MOD = [  # TODO if needed, we can add this to the input arguments
            ('t1', 'T1_nu.img'),
            ('fl', 'T1acq_nu_FL.img'),
            # ('annot', 'T1acq_nu_HfBd.img'),
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
    net.eval()

    # (4) loop over samples
    meters_test_set = list()
    for cnt, (image, annot, meta) in enumerate(test_loader):
        print(cnt+1, len(test_loader))
        meters = dict()

        image = image.to(device, non_blocking=True)
        annot = annot.to(device, non_blocking=True)

        # (A) Get prediction
        pred = net(image)

        # (B) Threshold prediction
        pred = binarize_pred(pred, binarize_threshold=cfg.TEST.BINARIZE_THRESHOLD)

        # (C) Save prediction
        if save_pred_flag:
            save_pred(pred, cfg.OUTPUT_DIR, meta[0]['sample_path'], *[image])

        # (D) Evaluate prediction
        if eval_pred_flag:
            eval_pred(pred, annot, meters, cfg)
            meters_test_set.append(meters)

    print('*** Done saving segmentation prediction for the test data.'
          '*** Results are saved at:')
    print(cfg.OUTPUT_DIR)

    if eval_pred_flag:
        print('\nEvaluation results on the test set is:')
        for k in meters_test_set[0].keys():
            print(k, np.array([i[k] for i in meters_test_set]).mean())
