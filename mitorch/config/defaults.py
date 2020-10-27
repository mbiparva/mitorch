#!/usr/bin/env python3

"""Configs."""
#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
from fvcore.common.config import CfgNode
import datetime
import socket

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# -----------------------------------------------------------------------------
# Project options
# -----------------------------------------------------------------------------
_C.PROJECT = CfgNode()

# Project Name
_C.PROJECT.NAME = ''

# Project Description
_C.PROJECT.DESCRIPTION = ''

# Root directory of project
_C.PROJECT.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory
_C.PROJECT.DATASET_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'dataset'))

# Model directory
_C.PROJECT.MODELS_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'lib', 'models'))

# Experiment directory
_C.PROJECT.EXPERIMENT_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'experiments'))

# Set meters to use for experimental evaluation
_C.PROJECT.METERS = ['loss', 'dice_coefficient', 'jaccard_index', 'hausdorff_distance', 'f1']

# Training, Validation, and Test Split Ratio
_C.PROJECT.TVSR = 0.80
_C.PROJECT.TSR = 0.0
_C.PROJECT.PA_INDICES = (False, True)[0]


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Train a single experiment or batch of hyper-parameter optimization experiments
_C.TRAIN.HPO = (False, True)[0]

# Dataset.
_C.TRAIN.DATASET = ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILhfb', 'TRAP', 'CAPTURE', 'TRACING')[3]

# Input Modalities
_C.TRAIN.IN_MOD = {
    'WMHSegmentationChallenge': [
        ('t1', 'T1.nii.gz'),
        ('fl', 'FLAIR.nii.gz'),
        ('annot', 'wmh.nii.gz'),
    ],
    'SRIBIL': [
        ('t1', 'T1_nu.nii.gz'),
        ('fl', 'T1acq_nu_FL.nii.gz'),
        ('annot', 'wmh_seg.nii.gz'),
    ],
    'SRIBILhfb': [
        ('t1', 't1.nii.gz'),
        ('fl', 'fl.nii.gz'),
        # ('t2', 't2.nii.gz'),
        ('annot', 'truth.nii.gz'),
        ],
    'TRAP': [[], [], []],  # just to imitate the typical behaviour for INPUT_CHANNELS
    'CAPTURE': [[], []],
    'TRACING': [[], []],
}[_C.TRAIN.DATASET]

if socket.gethostname() == 'cerveau.sri.utoronto.ca':
    _C.PROJECT.DATASET_DIR = '/data2/projects/mitorch_datasets'

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1

# Shuffle the training data, valid / test are never shuffled
_C.TRAIN.SHUFFLE = True

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs. < 1 will deactivate
_C.TRAIN.CHECKPOINT_PERIOD = 0

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""


# ---------------------------------------------------------------------------- #
# Validation options
# ---------------------------------------------------------------------------- #
_C.VALID = CfgNode()

# If True validate the model, else skip the validation.
_C.VALID.ENABLE = True

# Total mini-batch size
_C.VALID.BATCH_SIZE = 1


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = ('SRIBILhfbTest', 'LEDUCQTest', 'PPMITest', 'SRIBILTest')[3]

# Input Modalities
_C.TEST.IN_MOD = {
    'SRIBILhfbTest': [
        ('t1', 'T1_nu.nii.gz'),
        ('fl', 'T1acq_nu_FL.nii.gz'),
        ('annot', 'T1acq_nu_HfBd.nii.gz'),
        # ('t2', 'T1acq_nu_T2.nii.gz'),
    ],
    'LEDUCQTest': [
        ('t1', 'T1_nu.nii.gz'),
        ('fl', 'T1acq_nu_FL_Warped.nii.gz'),
        ('annot', 'T1acq_nu_HfB_pred_edit.nii.gz'),
        # ('t2', 'T1acq_nu_T2_Warped.nii.gz'),
    ],
    'PPMITest': [
        ('t1', 'T1_nu.img'),
        ('fl', 'T1acq_nu_FL.img'),
        ('annot', 'T1acq_nu_HfBd.img'),
        # ('t2', 'T1acq_nu_T2.img'),
    ],
    'SRIBILTest': [
        ('t1', 'T1_nu.nii.gz'),
        ('fl', 'T1acq_nu_FL.nii.gz'),
        ('annot', 'wmh_seg.nii.gz'),
    ],
}[_C.TEST.DATASET]

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Path to the checkpoint to load the network weights.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Path to the test data
_C.TEST.DATA_PATH = ""

# Single or Batch test mode
_C.TEST.BATCH_MODE = (False, True)[0]

_C.TEST.BINARIZE_THRESHOLD = 0.55


# # ---------------------------------------------------------------------------- #
# # Batch norm options
# # ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "unet3d"

# Model name
_C.MODEL.MODEL_NAME = ('Unet3D', 'NestedUnet3D')[0]

_C.MODEL.PROCESSING_MODE = ('2d', '3d')[1]

# Loss function.
_C.MODEL.LOSS_FUNC = ('CrossEntropyLoss', 'DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss', 'LovaszLoss')[1]

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 2 if _C.MODEL.LOSS_FUNC == 'CrossEntropyLoss' else 1

_C.MODEL.LOSS_AUG_WHL = (False, True)[0]
_C.MODEL.WHL_NUM_DEPTH_SHEETS = (2, 4, 8)[2]
_C.MODEL.WHL_SEG_THR = (0.12, 0.25, 0.5)[1]

_C.MODEL.IGNORE_INDEX = 255

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.3  # according to TF implementation

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Number of base filters to the network
_C.MODEL.N_BASE_FILTERS = 16

# Encoder depth
_C.MODEL.ENCO_DEPTH = 5

# Decoder depth
_C.MODEL.NUM_PRED_LEVELS = 3

# Number of input channels to the model
_C.MODEL.INPUT_CHANNELS = (len(_C.TEST.IN_MOD) - 1) if _C.TEST.ENABLE else (len(_C.TRAIN.IN_MOD) - 1)

# Model settings
_C.MODEL.SETTINGS = tuple({
    'Unet3D': CfgNode({}),
    'NestedUnet3D': CfgNode({
        'DEEP_SUPERVISION': (False, True)[1],
        'N_HOP_DENSE_SKIP_CONNECTION': 2,  # must be > 0, 1 means no dense-skip-connections
        'MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[3],
    }),
}.items())


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The spatial max side size of the input volume.
_C.DATA.MAX_SIDE_SIZE = 192

_C.DATA.MIN_SIDE = (False, True)[0]

_C.DATA.UNI_SCALE = (False, True)[1]

# The spatial crop size of the input volume.
_C.DATA.CROP_SIZE = (192-16*2, 192-16*1, 192)[0]  # [0]  # change it from int to tuple for non-iso patching
_C.DATA.CROP_SIZE_FACTOR = 0  # BOAX script will change it at init.

# The spatial crop scale of the input volume.
_C.DATA.CROP_SCALE = ((0.7, 1.0), (0.8, 1.0), (0.9, 1.0))[1]

# The mean value of the volume raw voxels across the T1, FLAIR, T2 channels.
# ATTENTION: Assumes the order of channels is always T1, FLAIR, T2.
# _C.DATA.MEAN = [0.18278566002845764, 0.1672040820121765]  # MAGED PREP - TODO add it to the init of dataset
_C.DATA.MEAN = [0.058173052966594696, 0.044205766171216965, 0.04969067499041557][:_C.MODEL.INPUT_CHANNELS]  # [1:3]

# The standard deviation value of the volume raw voxels across the above channels.
# _C.DATA.STD = [0.018310515210032463, 0.017989424988627434]  # MAGED PREP
_C.DATA.STD = [0.021794982254505157, 0.02334374189376831, 0.024663571268320084][:_C.MODEL.INPUT_CHANNELS]

_C.DATA.PADDING_MODE = ('mean', 'median', 'min', 'max')[0]

# Dataset enforces canonical orientation and diagonality upon loading nii volumes
_C.DATA.ENFORCE_NIB_CANONICAL = (False, True)[0]
_C.DATA.ENFORCE_DIAG = (False, True)[0]

# Data transformation pipeline experimentation
_C.DATA.EXP = CfgNode()

_C.DATA.EXP.HEAD_ORI = (0, 1)[0]
_C.DATA.EXP.HEAD_RES = (0, 1)[0]
_C.DATA.EXP.BODY_CRO = (0, 1, 2, 3)[0]
_C.DATA.EXP.BODY_FLI = (0, 1)[0]
_C.DATA.EXP.INTENSITY = (False, True)[0]
_C.DATA.EXP.INTENSITY_SEL = (0, 1, 2, 3, 4, 5, 6, 7, 8)[1]


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 1e-3

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 100

# Momentum.
_C.SOLVER.MOMENTUM = 0.8

# Dampening for Momentum
_C.SOLVER.DAMPENING = 0

# Nesterov momentum.
_C.SOLVER.NESTEROV = False

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-5

# TODO add warm-up and scheduler hypers here

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = ('sgd', 'adadelta', 'adagrad', 'rmsprop', 'adam')[4]

# Enable Scheduler
_C.SOLVER.SCHEDULER_MODE = False

# Set the type of scheduler
_C.SOLVER.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau', 'cosine')[-2]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Use GPUs
_C.USE_GPUS = (False, True)[1]

# Default GPU device id
_C.GPU_ID = 0

# Whether to use auto-mixed-precision (amp)
_C.AMP = _C.USE_GPUS and (False, True)[0]

# Whether to use DataParallel in PyTorch
_C.DATA_PARALLEL = _C.USE_GPUS and (False, True)[0]  # uses all gpus on the device

# Whether to use DistributedDataParallel in PyTorch
_C.DISTRIBUTED_DATA_PARALLEL = _C.USE_GPUS and (False, True)[0]  # uses all gpus on the device

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 110

# Log period in iters.
_C.LOG_PERIOD = 2


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Hyper-Parameter Optimization
# ---------------------------------------------------------------------------- #
_C.HPO = CfgNode()

_C.HPO.MODE = ('MAN', 'BOAX')[1]

_C.HPO.MAX_EPOCH = 200

# Starting element of the HPO range --- MUST BE SET
_C.HPO.RANGE_START = 0

# length of the HPO range --- MUST BE SET
_C.HPO.RANGE_LEN = 0

_C.HPO.EVAL_METRIC = _C.PROJECT.METERS[1]

_C.HPO.TOTAL_TRIALS = 100


# # ---------------------------------------------------------------------------- #
# # WMH options
# # ---------------------------------------------------------------------------- #
_C.WMH = CfgNode()

# Whether WMH is enabled
_C.WMH.ENABLE = True if _C.TRAIN.DATASET in ('WMHSegmentationChallenge', 'SRIBIL') else False

# HFB checkpoint to load
_C.WMH.HFB_CHECKPOINT = os.path.join(_C.PROJECT.EXPERIMENT_DIR,
                                     'SRIBILhfb/20200612_134356_471570/checkpoints/checkpoint_epoch_00060.pyth')

# Whether to use ground-truth HFB masks
_C.WMH.HFB_GT = (False, True)[1] if _C.TRAIN.DATASET == 'SRIBIL' else False  # must be True only when SRIBIL

_C.WMH.BINARIZE_THRESHOLD = 0.5

_C.WMH.MAX_SIDE_SIZE = 160

_C.WMH.FILL = 0

_C.WMH.PADDING_MODE = ('mean', 'median', 'min', 'max')[0]

_C.WMH.CROPPING = (False, True)[0]
_C.WMH.RESIZING_PADDING = (False, True)[0]


# # ---------------------------------------------------------------------------- #
# # Neuron and Virus Tracing Segmentation options
# # ---------------------------------------------------------------------------- #
_C.NVT = CfgNode()

# Whether NVT is enabled
_C.NVT.ENABLE = True if _C.TRAIN.DATASET in ('TRAP', 'CAPTURE', 'TRACING') else False

_C.NVT.NUM_MULTI_PATCHES = 1

_C.NVT.PATCH_SELECTION_POLICY = (False, True)[0]

_C.NVT.ENFORCE_SELECTION_POLICY = (False, True)[0]

_C.NVT.SELECTION_LB = 0

_C.NVT.RANDOM_CROP_NUM_ATTEMPS = 20
_C.NVT.RANDOM_CROP_THRESHOLD = 0

_C.NVT.REPEAT_DATASET = 0  # < 2 is off


def init_cfg(cfg, parent_dir=''):
    """ Initialize those with hierarchical dependencies and conditions, critical for multi-case experimentation"""
    # Model ID
    cfg.MODEL.ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Output basedir.
    destin_set = cfg.TEST.DATASET if cfg.TEST.ENABLE else cfg.TRAIN.DATASET
    cfg.OUTPUT_DIR = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, destin_set, parent_dir, cfg.MODEL.ID)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    return cfg


def _assert_and_infer_cfg(cfg):
    # TODO add assertion respectively
    if 'N_HOP_DENSE_SKIP_CONNECTION' in cfg.MODEL.SETTINGS:
        assert cfg.MODEL.SETTINGS.N_HOP_DENSE_SKIP_CONNECTION > 0

    assert not (cfg.DATA_PARALLEL and cfg.DISTRIBUTED_DATA_PARALLEL), 'either use DP or DDP in PyTorch'

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    cfg = _C.clone()
    cfg = init_cfg(cfg)
    cfg = _assert_and_infer_cfg(cfg)
    return cfg
