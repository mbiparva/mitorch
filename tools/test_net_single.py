#!/usr/bin/env python3

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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
import utils.checkpoint as checkops
from data.build import build_dataset
from config.defaults import init_cfg
from netwrapper.net_wrapper import NetWrapperHFB, NetWrapperWMH
from datetime import datetime
import logging
import pandas as pd
import utils.metrics as metrics


def setup_logger():
    local_logger = logging.getLogger(__name__)
    local_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler if you wish
    file_handler = logging.FileHandler('/tmp/test_error_output_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M')))
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    # create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    local_logger.addHandler(file_handler)
    local_logger.addHandler(stream_handler)

    return local_logger


logger = setup_logger()


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


def eval_pred(p, a, meters, cfg):
    for m in cfg.TEST.EVAL_METRICS:
        metric_function = getattr(metrics, f'{m}_metric')
        meters[m] = metric_function(p, a, ignore_index=cfg.MODEL.IGNORE_INDEX)


def reset_cfg_init(cfg):
    if not len(os.listdir(cfg.OUTPUT_DIR)):
        os.rmdir(cfg.OUTPUT_DIR)  # the old one is useless, we create a new one instead
    cfg = init_cfg(cfg)

    return cfg


def setup_test(cfg):
    cfg = reset_cfg_init(cfg)

    cfg.TRAIN.ENABLE = cfg.VALID.ENABLE = False
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cuda_device_id = cfg.GPU_ID
    torch.cuda.set_device(cuda_device_id)
    if cfg.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(cuda_device_id))
        logger.info('cuda available')
        logger.info(f'device count is {torch.cuda.device_count()}')
        logger.info(f'{device} will be used ...')
    else:
        device = torch.device('cpu')

    assert cfg.TEST.CHECKPOINT_FILE_PATH and \
        len(cfg.TEST.CHECKPOINT_FILE_PATH) and \
        os.path.exists(cfg.TEST.CHECKPOINT_FILE_PATH), 'TEST.CHECKPOINT_FILE_PATH not set'

    return cfg, device


def build_transformations():
    transformations = torch_tf.Compose([
        tf.ToTensorImageVolume(),
        tf.RandomOrientationTo('RPI'),
        # tf.RandomResampleTomm(target_spacing=(1, 1, 1)),
        tf.ConcatAnnot2ImgVolume(num_channels=-1),  # concat all except the last to the image
        tf.MaskIntensityVolume(mask_data=None),  # crop a tight 3D box
        tf.ConcatAnnot2ImgVolume(num_channels=-1),  # concat all annot to the image
        tf.CropForegroundVolume(margin=1),  # crop the brain region
        tf.ConcatImg2AnnotVolume(num_channels=2),
        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
    ])

    return transformations


def create_test_set(cfg, transformations):
    # Define any test dataset with annotation as known dataset otherwise call TestSet
    if len(cfg.TEST.DATA_PATH):
        if cfg.WMH.ENABLE and not cfg.WMH.HFB_GT:
            assert cfg.WMH.HFB_CHECKPOINT and \
                len(cfg.WMH.HFB_CHECKPOINT) and \
                os.path.exists(cfg.WMH.HFB_CHECKPOINT), 'WMH.HFB_CHECKPOINT not set'

        logger.info('you chose {} mode'.format(('single', 'batch')[cfg.TEST.BATCH_MODE]))
        eval_pred_flag = (False, True)[1]
        save_pred_flag = True

        cfg.TEST.DATASET = 'TestSet{}'.format(('single', 'batch')[cfg.TEST.BATCH_MODE].upper())
        cfg = reset_cfg_init(cfg)

        cfg.TEST.IN_MOD = [  # TODO if needed, we can add this to the input arguments
            # ('t1', 'T1_nu.img'),
            # ('fl', 'T1acq_nu_FL.img'),
            # ('annot', 'T1acq_nu_HfBd.img'),
            # ('t1', 'T1_nu.nii.gz'),  # wmh test cases
            # ('fl', 'T1acq_nu_FL.nii.gz'),
            # ('annot', 'wmh_seg.nii.gz'),
            ('t1', 'T1.nii.gz'),  # wmh challenge test cases
            ('fl', 'FLAIR.nii.gz'),
            ('annot', 'wmh.nii.gz'),
        ]
        test_set = TestSet(cfg, 'test', transformations, prefix_name=False if cfg.WMH.ENABLE else True)
    else:
        test_set = build_dataset(cfg.TEST.DATASET, cfg, 'test', transformations)

    return test_set


def create_net(cfg, device):
    if cfg.WMH.ENABLE:
        net_wrapper = NetWrapperWMH(device, cfg)
    else:
        net_wrapper = NetWrapperHFB(device, cfg)

    checkops.load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, net_wrapper.net_core, distributed_data_parallel=False)
    net_wrapper.net_core.eval()

    return net_wrapper


def test_loop(cfg, test_loader, device, net_wrapper, save_pred_flag, eval_pred_flag):
    meters_test_set = list()
    for cnt, (image, annot, meta) in enumerate(test_loader):
        logger.info(f'testing on: {cnt+1:05d}|{len(test_loader):05d}')
        meters = dict()

        image = image.to(device, non_blocking=True)
        annot = annot.to(device, non_blocking=True)

        # (A) Get prediction
        if cfg.WMH.ENABLE:
            pred, annot, image = net_wrapper.forward((image, annot), return_input=True)
        else:
            pred = net_wrapper.forward(image)

        # (B) Threshold prediction
        pred = binarize_pred(pred, binarize_threshold=cfg.TEST.BINARIZE_THRESHOLD)

        # (C) Save prediction
        if save_pred_flag:
            save_pred(pred, cfg.OUTPUT_DIR, meta[0]['sample_path'], *[image])

        # (D) Evaluate prediction
        if eval_pred_flag:
            eval_pred(pred, annot, meters, cfg)
            meters_test_set.append(meters)

    if save_pred_flag:
        logger.info('*** Done saving segmentation prediction for the test data.'
                    '*** Results are saved at: {}'.format(cfg.OUTPUT_DIR))

    return meters_test_set


def get_output_results(meters_test_set, eval_pred_flag):
    output_results = dict()
    if eval_pred_flag:
        logger.info('Evaluation results on the test set is ---')
        meters_test_set = pd.DataFrame(meters_test_set)
        meters_mean = meters_test_set.mean()
        meters_std = meters_test_set.std()
        for k in meters_test_set.columns:
            output_results[f'{k}_mean'] = meters_mean[k]
            output_results[f'{k}_std'] = meters_std[k]

        logger.info(output_results)

    return output_results


@torch.no_grad()
def test(cfg, transformations=None, eval_pred_flag=True, save_pred_flag=True):

    # (0) initial setup
    cfg, device = setup_test(cfg)

    # (1) define data pipeline
    transformations = build_transformations() if transformations is None else transformations

    # (2) create test set and loader
    test_set = create_test_set(cfg, transformations)

    test_loader = DataLoader(test_set,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=False,
                             drop_last=False,
                             num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                             pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                             worker_init_fn=ds_worker_init_fn,
                             collate_fn=collate_fn,
                             )

    # (3) create network and load snapshots
    net_wrapper = create_net(cfg, device)

    # (4) loop over samples
    meters_test_set = test_loop(cfg, test_loader, device, net_wrapper, save_pred_flag, eval_pred_flag)

    # (5) log formatted outputs
    output_results = get_output_results(meters_test_set, eval_pred_flag)

    return output_results
