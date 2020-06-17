#!/usr/bin/env python3

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import nibabel as nib
import data.transforms_mitorch as tf
from torch.utils.data import DataLoader
import torchvision.transforms as torch_tf
from data.data_container import ds_worker_init_fn
from data.SRIBILSet import SRIBIL
from data.VolSet import collate_fn
from models.build import build_model
import utils.checkpoint as checkops
import torch
import numpy as np


class TestSet(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)

    def _init_dataset(self):
        self.dataset_path = self.dataset_root
        self.in_modalities = {
            't1': 'T1.img',
            'fl': 'FL.img',
        }
        self.sample_path_list = self.index_samples()

    def index_samples(self):
        return [self.cfg.TEST.DATA_PATH] if not self.cfg.TEST.BATCH_MODE else [
            os.path.join(self.cfg.TEST.DATA_PATH, i)
            for i in sorted(os.listdir(self.cfg.TEST.DATA_PATH))
            ]

    @staticmethod
    def put_fname_template(path_name, file_name):
        par_dir = path_name.rpartition('/')[-1]
        return '{}_{}'.format(
            par_dir,
            file_name
        )

    def find_data_files_path(self, sample_path):
        return {
            u: os.path.join(sample_path, self.put_fname_template(sample_path, v))
            for u, v in self.in_modalities.items()
        }

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        in_pipe_data = self.find_data_files_path(sample_path)
        in_pipe_data = self.load_data(in_pipe_data,
                                      enforce_nib_canonical=self.cfg.DATA.ENFORCE_NIB_CANONICAL,
                                      enforce_diag=self.cfg.DATA.ENFORCE_DIAG,
                                      dtype=np.float32)
        in_pipe_data, in_pipe_meta = self.extract_data_meta(in_pipe_data)

        in_pipe_meta = self.run_sanity_checks(in_pipe_meta)
        in_pipe_meta['sample_path'] = sample_path

        image_tensor = self.get_data_tensor(in_pipe_data)
        annot_tensor = torch.zeros_like(image_tensor)

        image_tensor = torch.stack(image_tensor, dim=-1)  # D x H x W x C
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, _, in_pipe_meta = self.transform((image_tensor, annot_tensor, in_pipe_meta))

        return image_tensor, in_pipe_meta


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
    test_set = TestSet(cfg, 'test', transformations)
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
    for cnt, (image, meta) in enumerate(test_loader):
        print(cnt, len(test_loader))
        image = image.to(device, non_blocking=True)

        # (C) Get prediction
        pred = net(image)

        # (D) Threshold prediction
        pred = binarize_pred(pred, binarize_threshold=cfg.TEST.BINARIZE_THRESHOLD)

        # (E) Save prediction
        save_pred(pred, cfg.OUTPUT_DIR)

    print('*** Done saving segmentation prediction for the test data.'
          '*** Results are saved at:')
    print(cfg.OUTPUT_DIR)
