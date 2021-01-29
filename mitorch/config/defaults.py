#!/usr/bin/env python3

"""Configs."""
#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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

# Dataset list
_C.PROJECT.KNOWN_DATASETS = ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILhfb',
                             'TRAP', 'CAPTURE', 'TRACING', 'TRACINGSEG', 'HPSubfield')


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Train a single experiment or batch of hyper-parameter optimization experiments
_C.TRAIN.HPO = (False, True)[0]

# Dataset for training
_C.TRAIN.DATASET = ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILhfb',
                    'TRAP', 'CAPTURE', 'TRACING', 'TRACINGSEG',
                    'HPSubfield')[1]

# Input Modalities
_C.TRAIN.IN_MOD = tuple({
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
    'TRACINGSEG': [[], []],
    'TRACING': [[], []],
    'HPSubfield': [
        ('t1', 't1.nii.gz'),
        ('t2', 't2.nii.gz'),
        ('annot', 'truth.nii.gz'),
    ],
}.items())

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

# binarization threshold used for performance evaluation
_C.TRAIN.BINARIZE_THRESHOLD = 0.50


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
_C.TEST.IN_MOD = tuple({
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
}.items())

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Path to the checkpoint to load the network weights.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Path to the test data
_C.TEST.DATA_PATH = ""

# Single or Batch test mode
_C.TEST.BATCH_MODE = (False, True)[0]

_C.TEST.BINARIZE_THRESHOLD = 0.55

# whether to run the set of robustness experiments
_C.TEST.ROBUST_EXP = (False, True)[0]

# if so, loop over the list of experiments each having transformations
_C.TEST.ROBUST_EXP_LIST = []

_C.TEST.EVAL_METRICS = ['dice_coefficient', 'jaccard_index', 'hausdorff_distance', 'rvd']

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
_C.MODEL.MODEL_NAME = ('Unet3D', 'NestedUnet3D', 'Unet3DCBAM', 'DAUnet3D',
                       'MBUnet3D', 'MUnet3D', 'DYNUnet3D', 'Vnet3D', 'Densenet3D', 'Highresnet3D', 'SEUnet3D')[0]

_C.MODEL.PROCESSING_MODE = ('2d', '3d')[1]

# Loss function.
_C.MODEL.LOSSES = [
    {
        'name': ('CrossEntropyLoss', 'DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss', 'LovaszLoss', 'MSELoss')[1],
        'weight': 1.0,
        'with_logits': True,
    },
    # {
    #     'name': ('CrossEntropyLoss', 'DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss', 'LovaszLoss', 'MSELoss')[3],
    #     'weight': 0.1,
    #     'with_logits': True,
    # },
    # {
    #     'name': ('CrossEntropyLoss', 'DiceLoss', 'WeightedHausdorffLoss', 'FocalLoss', 'LovaszLoss', 'MSELoss')[5],
    #     'weight': 0.1,
    #     'with_logits': True,
    # },
]

# Loss functions that would need logits separately.
_C.MODEL.LOSSES_WITH_LOGITS = ('DiceLoss', 'WeightedHausdorffLoss', 'MSELoss', )

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = -1

# Weighted Hausdorff loss options
_C.MODEL.WHL_NUM_DEPTH_SHEETS = (2, 4, 8)[2]
_C.MODEL.WHL_SEG_THR = (0.12, 0.25, 0.5)[1]

# Ignore index; < 0 values disable it
_C.MODEL.IGNORE_INDEX = 255

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.3  # according to TF implementation

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Encoder depth
_C.MODEL.ENCO_DEPTH = 5

# Decoder depth
_C.MODEL.NUM_PRED_LEVELS = 3

# Number of input channels to the model
_C.MODEL.INPUT_CHANNELS = 0  # will set later in the init_dependencies

# Model settings
_C.MODEL.SETTINGS = CfgNode({
    'Unet3D': CfgNode({
        'N_BASE_FILTERS': 16,
        'ENCODER_STRIDE': (2, 2, 2),
        'ENCODER_DILATION': (2, 2, 2),
        'DECODER_DILATION': (2, 2, 2),
    }),
    'NestedUnet3D': CfgNode({
        'N_BASE_FILTERS': 16,
        'ENCODER_STRIDE': (2, 2, 2),
        'ENCODER_DILATION': (2, 2, 2),
        'DEEP_SUPERVISION': (False, True)[1],
        'N_HOP_DENSE_SKIP_CONNECTION': 2,  # must be > 0, 1 means no dense-skip-connections
        'MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[3],
    }),
    'Unet3DCBAM': CfgNode({
        'N_BASE_FILTERS': 16,
        'ENCODER_STRIDE': (2, 2, 2),
        'ENCODER_DILATION': (2, 2, 2),
        'DECODER_DILATION': (2, 2, 2),
        'GAM': CfgNode({
            'BLOCKS': [3, 4],
            'REDUCTION_RATIO': 16,
            'CROSS_MODAL_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[0],
            'REF_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[1],
            'RESIDUAL': (False, True)[1],
            'NUM_CONV_BLOCKS': 2,
            'DILATION': 4,
            'CHANNEL_POOLING_TYPE': ('max', 'average', 'pa', 'lse')[1],
            'CHANNEL': (False, True)[1],
            'SPATIAL': (False, True)[1],
            'SPATIAL_KERNEL_SIZE': 3,
            'NORM_TYPE': ('batch', 'instance', 'layer', 'none')[0],
        }),
        'LAM': CfgNode({
            'BLOCKS': [3, 4],
            'REDUCTION_RATIO': 16,
            'RESIDUAL_RELATIVE': ('before', 'after')[0],
            'SPATIAL_POOLING_TYPE': ('max', 'average', 'max_average')[2],
            'CHANNEL_POOLING_TYPES': ['max', 'average', 'pa', 'lse'][:2],
            'CHANNEL_TYPES_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[0],
            'CHANNEL_POOLING_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[3],
            'ATTENTION_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[1],
            'REF_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation', 'bypass')[3],
            'RESIDUAL': (False, True)[1],
            'CHANNEL': (False, True)[1],
            'SPATIAL': (False, True)[1],
            'SPATIAL_KERNEL_SIZE': 3,
        }),
    }),
    'DAUnet3D': CfgNode({
        'N_BASE_FILTERS': 16,
        'ENCODER_STRIDE': (2, 2, 2),
        'ENCODER_DILATION': (2, 2, 2),
        'DECODER_DILATION': (2, 2, 2),
        'GAM': CfgNode({
            'BLOCKS': [-1],  # switches off Global Attention Module for DANet
        }),
        'LAM': CfgNode({
            'BLOCKS': [3, 4],
            'RESIDUAL_RELATIVE': ('before', 'after')[0],
            'INPUT_REDUCTION_RATIO': 4,
            'MIDDLE_REDUCTION_RATIO': 8,
            'INTERNAL_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[0],
            'CROSS_MODAL_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[0],
            'REF_MODULATION_TYPE': ('additive', 'multiplicative', 'mean', 'concatenation')[1],
            'RESIDUAL': (False, True)[1],
            'CHANNEL': (False, True)[1],
            'SPATIAL': (False, True)[1],
            'KERNEL_SIZE': 3,
        }),
    }),
    'MBUnet3D': CfgNode({
        'FEATURES': (32, 32, 64, 128, 256, 32),
        'ACT': ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        'NORM': ("instance", {"affine": True}),
        'UPSAMPLE': ("deconv", "nontrainable")[1],
    }),
    'MUnet3D': CfgNode({
        'CHANNELS': (32, 32, 64, 128, 256),
        'STRIDES': (2, 2, 2, 2, 2),
        'KERNEL_SIZE': 3,
        'UP_KERNEL_SIZE': 3,
        'NUM_RES_UNITS': 0,
        'ACT': ('relu', 'elu', 'leakyrelu', 'prelu', 'relu6', 'selu', 'celu', 'gelu', 'swish', 'mish')[2],
        'NORM': ('instance', 'batch', 'group', 'layer')[0],  # we can pass args like MBUnet3D
    }),
    'Densenet3D': CfgNode({
        '_NET_SEL': (0, 1, 2, 3)[0],
        'BN_SIZE': (3, 4)[0],
        'INIT_FEATURES': 16,
        'GROWTH_RATE': 8,
        'DECODER_DILATION': (2, 2, 2),
    }),
    'Highresnet3D': CfgNode({
        'NORM_TYPE': ('batch', 'instance')[1],
        'ACTI_TYPE': ('relu', 'prelu', 'relu6')[0],
        'DECODER_DILATION': (2, 2, 2),
    }),
    'DYNUnet3D': CfgNode({
        'KERNEL_SIZE': (3, 3, 3, 3, 3),
        'STRIDES': (1, 2, 2, 2, 2),
        'UPSAMPLE_KERNEL_SIZE': (2, 2, 2, 2),  # one size less than kernel size tuple
        'NORM_NAME': ('instance', 'batch', 'group', 'layer')[0],
        'DEEP_SUPERVISION': True,
        'DEEP_SUPR_NUM': 2,
        'RES_BLOCK': True,
    }),
    'SEUnet3D': CfgNode({
        '_NET_SEL': (0, 1, 2, 3, 4, 5)[0],
        'DECODER_DILATION': (2, 2, 2),
    }),
    'Vnet3D': CfgNode({
        'ACT': ('relu', 'elu', 'leakyrelu', 'prelu', 'relu6', 'selu', 'celu', 'gelu', 'swish', 'mish')[2],
    }),
})


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
_C.DATA.MEAN = [0.058173052966594696, 0.044205766171216965, 0.04969067499041557]

# The standard deviation value of the volume raw voxels across the above channels.
_C.DATA.STD = [0.021794982254505157, 0.02334374189376831, 0.024663571268320084]

_C.DATA.PADDING_MODE = ('mean', 'median', 'min', 'max')[0]

# Dataset enforces canonical orientation and diagonality upon loading nii volumes
_C.DATA.ENFORCE_NIB_CANONICAL = (False, True)[0]
_C.DATA.ENFORCE_DIAG = (False, True)[0]

# Data transformation pipeline experimentation
_C.DATA.EXP = CfgNode()

_C.DATA.EXP.HEAD_ORI = (0, 1)[0]
_C.DATA.EXP.HEAD_RES = (0, 1)[0]
_C.DATA.EXP.BODY_CRO = (0, 1, 2, 3)[0]
_C.DATA.EXP.BODY_FLI = (0, 1, 2, 3, 4)[0]
_C.DATA.EXP.INTENSITY = (False, True)[0]
_C.DATA.EXP.INTENSITY_SEL = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)[1]
_C.DATA.EXP.TAIL_NORM = (0, 1, 2)[0]


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
_C.USE_GPU = (False, True)[1]

# Default GPU device id
_C.GPU_ID = 0

# Whether to use auto-mixed-precision (amp)
_C.AMP = _C.USE_GPU and (False, True)[0]

# Whether to use DataParallel in PyTorch
_C.DP = _C.USE_GPU and (False, True)[0]  # uses all gpus on the device

# Whether to use DistributedDataParallel in PyTorch
_C.DDP = _C.USE_GPU and (False, True)[0]  # uses all gpus on the device

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 110

# Log period in iters.
_C.LOG_PERIOD = 2


# ---------------------------------------------------------------------------- #
# DDP options
# ---------------------------------------------------------------------------- #
_C.DDP_CFG = CfgNode()

# Global rank of the process
_C.DDP_CFG.RANK = -1

# Local rank of the process
_C.DDP_CFG.LOCAL_RANK = -1

# World size of the DDP
_C.DDP_CFG.WORLD_SIZE = -1

# Number of nodes
_C.DDP_CFG.NNODES = -1

# Number of processes per node == number of GPUs per node
_C.DDP_CFG.NPROC_PER_NODE = -1


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


# ---------------------------------------------------------------------------- #
# WMH options
# ---------------------------------------------------------------------------- #
_C.WMH = CfgNode()

# Whether WMH is enabled
_C.WMH.ENABLE = _C.TRAIN.DATASET in ('WMHSegmentationChallenge', 'SRIBIL')

# HFB checkpoint to load
_C.WMH.HFB_CHECKPOINT = os.path.join(_C.PROJECT.EXPERIMENT_DIR,
                                     'SRIBILhfb/20200612_134356_471570/checkpoints/checkpoint_epoch_00060.pyth')

# Whether to use ground-truth HFB masks
_C.WMH.HFB_GT = _C.TRAIN.DATASET == 'SRIBIL' and (False, True)[1]  # must be True only when SRIBIL

_C.WMH.HFB_MASKING_MODE = ['manual', 'pipeline'][0]

_C.WMH.BINARIZE_THRESHOLD = 0.5

_C.WMH.MAX_SIDE_SIZE = 160

_C.WMH.FILL = 0

_C.WMH.PADDING_MODE = ('mean', 'median', 'min', 'max')[0]

_C.WMH.CROPPING = (False, True)[0]
_C.WMH.RESIZING_PADDING = (False, True)[0]


# ---------------------------------------------------------------------------- #
# Neuron and Virus Tracing Segmentation options
# ---------------------------------------------------------------------------- #
_C.NVT = CfgNode()

# Whether NVT is enabled
_C.NVT.ENABLE = _C.TRAIN.DATASET in ('TRAP', 'CAPTURE', 'TRACINGSEG', 'TRACING')

_C.NVT.NUM_MULTI_PATCHES = 16

_C.NVT.PATCH_SELECTION_POLICY = (False, True)[1]

_C.NVT.ENFORCE_SELECTION_POLICY = (False, True)[0]

_C.NVT.SELECTION_LB = 16000

_C.NVT.RANDOM_CROP_NUM_ATTEMPS = 500
_C.NVT.RANDOM_CROP_THRESHOLD = 0

_C.NVT.REPEAT_DATASET = 0  # < 2 is off

_C.NVT.BINARY_SEG = (False, True)[1]


# ---------------------------------------------------------------------------- #
# Hippocampus Subfield Segmentation options
# ---------------------------------------------------------------------------- #
_C.HPSF = CfgNode()

# Whether NVT is enabled
_C.HPSF.ENABLE = _C.TRAIN.DATASET in ('HPSubfield', )


# noinspection PyTypeChecker
def init_dependencies(cfg):
    assert cfg.TRAIN.DATASET in _C.PROJECT.KNOWN_DATASETS, 'dataset is unknown'
    
    cfg.MODEL.SETTINGS = cfg.MODEL.SETTINGS[cfg.MODEL.MODEL_NAME] if cfg.MODEL.MODEL_NAME in cfg.MODEL.SETTINGS else _C.MODEL.SETTINGS[cfg.MODEL.MODEL_NAME]

    cfg.TRAIN.IN_MOD = dict(_C.TRAIN.IN_MOD)[cfg.TRAIN.DATASET]

    cfg.TEST.IN_MOD = dict(_C.TEST.IN_MOD)[cfg.TEST.DATASET]

    if cfg.MODEL.INPUT_CHANNELS <= 0:  # if not set in the yaml cfg file
        cfg.MODEL.INPUT_CHANNELS = (len(cfg.TRAIN.IN_MOD) - 1, len(cfg.TEST.IN_MOD) - 1)[cfg.TEST.ENABLE is True]

    cfg.DATA.MEAN = _C.DATA.MEAN[:cfg.MODEL.INPUT_CHANNELS]
    cfg.DATA.STD = _C.DATA.MEAN[:cfg.MODEL.INPUT_CHANNELS]

    cfg.WMH.ENABLE = cfg.TRAIN.DATASET in ('WMHSegmentationChallenge', 'SRIBIL')
    cfg.WMH.HFB_GT = cfg.TRAIN.DATASET == 'SRIBIL' and cfg.WMH.HFB_GT
    if cfg.WMH.ENABLE and cfg.TRAIN.BATCH_SIZE == 1:  # if batch size is one, batchnorm throws error, hence instance
        cfg.MODEL.SETTINGS.Unet3DCBAM.GAM.NORM_TYPE = 'layer'

    cfg.NVT.ENABLE = cfg.TRAIN.DATASET in ('TRAP', 'CAPTURE', 'TRACINGSEG', 'TRACING')
    if cfg.NVT.ENABLE and cfg.NVT.BINARY_SEG:
        cfg.MODEL.NUM_CLASSES = 1

    cfg.HPSF.ENABLE = cfg.TRAIN.DATASET in ('HPSubfield',)

    for loss in cfg.MODEL.LOSSES:
        loss['with_logits'] = loss['name'] in cfg.MODEL.LOSSES_WITH_LOGITS and loss['with_logits']

    if not cfg.HPSF.ENABLE:
        cfg.MODEL.NUM_CLASSES = 2 if cfg.MODEL.LOSSES[0]['name'] == 'CrossEntropyLoss' else 1

    cfg.MODEL.SETTINGS = cfg.MODEL.SETTINGS[cfg.MODEL.MODEL_NAME] if cfg.MODEL.MODEL_NAME in cfg.MODEL.SETTINGS \
        else CfgNode({})

    return cfg


def init_cfg(cfg, parent_dir=''):
    """ Initialize those with hierarchical dependencies and conditions, critical for multi-case experimentation"""
    # Model ID
    cfg.MODEL.ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Output basedir.
    destin_set = cfg.TEST.DATASET if cfg.TEST.ENABLE else cfg.TRAIN.DATASET
    cfg.OUTPUT_DIR = os.path.join(cfg.PROJECT.EXPERIMENT_DIR, destin_set, parent_dir, cfg.MODEL.ID)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    cfg = init_dependencies(cfg)

    cfg = _assert_and_infer_cfg(cfg)

    return cfg


def _assert_and_infer_cfg(cfg):
    # TODO add assertion respectively
    if 'N_HOP_DENSE_SKIP_CONNECTION' in cfg.MODEL.SETTINGS:
        assert cfg.MODEL.SETTINGS.N_HOP_DENSE_SKIP_CONNECTION > 0

    assert not (cfg.DP and cfg.DDP), 'either use DP or DDP in PyTorch'

    assert len(cfg.MODEL.LOSSES) > 0, 'at least one loss must be defined'

    return cfg


def get_cfg(delayed_init=False):
    """
    Get a copy of the default config.
    """
    cfg = _C.clone()
    if not delayed_init:  # needed for DDP
        cfg = init_cfg(cfg)
    cfg = _assert_and_infer_cfg(cfg)
    return cfg
