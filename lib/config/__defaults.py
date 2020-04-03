"""Config file setting hyperparameters

This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

from easydict import EasyDict as edict
import os
import datetime
import socket

__C = edict()
cfg = __C   # from defaults.py import cfg


# ================
# GENERAL
# ================

# Set modes
__C.TRAINING = True
__C.VALIDATING = True

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory
__C.DATASET_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'dataset'))

# Model directory
__C.MODELS_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'lib', 'models'))

# Experiment directory
__C.EXPERIMENT_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'experiment'))

# Set meters to use for experimental evaluation
__C.METERS = ['loss', 'label_accuracy']

# Use GPU
__C.USE_GPU = True

# Default GPU device id
__C.GPU_ID = 0

# Number of epochs
__C.NUM_EPOCH = 100

# Number of workers
__C.NUM_WORKERS = 6

# Dataset name
__C.DATASET_NAME = ('LCTv0', 'LCTv1')[1]

# Crop Mode
__C.CROP_MODE = ('x1', 'x2', 'x3', 'x4')[0]

# End Frame Offset
__C.EFOFFSET = 10  # 1 sec @ 10HZ
__C.tTTE = (20, 30, 40)[0]
__C.tTTE_OFFSET = (0, 10, 20)[0]
# Number of categories
__C.NUM_CLASSES = 3

__C.DATASET_ROOT = os.path.join(__C.DATASET_DIR, __C.DATASET_NAME)

# Normalize database samples according to some mean and std values
__C.DATASET_NORM = True

# Input data size
__C.SPATIAL_INPUT_SIZE = (112, 112)
__C.CHANNEL_INPUT_SIZE = 3
__C.TEMPORAL_INPUT_SIZE = 8

# Training, Validation, and Test Split Ratio
__C.TVSR = 0.80
__C.TSR = 0.20

# Set parameters for snapshot and verbose routines
__C.MODEL_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
__C.SNAPSHOT = True
__C.SNAPSHOT_INTERVAL = 1
__C.VERBOSE = True
__C.VERBOSE_INTERVAL = 10
__C.VALID_INTERVAL = 1

# Network Architecture
__C.NET_ARCH = ('resnet', )[0]

# Pre-trained network
__C.PRETRAINED_MODE = (None, 'Custom')[0]

# Path to the pre-segmentation network
__C.PT_PATH = os.path.join(__C.EXPERIMENT_DIR, 'snapshot', '20181010_124618_219443', '079.pt')

# Pre-trained network
__C.PRETRAINED_MODE = (None, 'ImageNet', 'ResNet_ST', 'Custom')[0]

# =============================
# Spatiotemporal ResNet options
# =============================
__C.RST = edict()

__C.RST.CROSS_STREAM_MOD_LAYER = 2
__C.RST.TEMPORAL_CONVOLUTION_LAYER = -1  # turn it off with -1
__C.RST.INIT_TEMPORAL_STRATEGY = ('center', 'difference', 'average')[0]
__C.RST.VALID_F25 = False
__C.RST.LR_S_STREAM_MULT = 0.75

__C.FRAME_SAMPLING_METHOD = ('uniform', 'temporal_stride', 'random', 'temporal_stride_random')[0]
__C.NFRAMES_PER_VIDEO = 20  # T x tau
__C.TEMPORAL_STRIDE = (1, 25)
__C.FRAME_RANDOMIZATION = False

# # =============================
# # SlowFast ResNet options
# # =============================
# __C.SLOWFAST = edict()
#
# # T = NFRAMES_PER_VIDEO // TAU
# __C.SLOWFAST.TAU = 16
# __C.SLOWFAST.ALPHA = 8
# __C.SLOWFAST.T2S_MUL = 2
# __C.SLOWFAST.DP = 0.5

# ================
# Training options
# ================
if __C.TRAINING:
    __C.TRAIN = edict()

    # Images to use per minibatch
    __C.TRAIN.BATCH_SIZE = 32

    # Shuffle the dataset
    __C.TRAIN.SHUFFLE = True

    # Learning parameters are set below
    __C.TRAIN.LR = 1e-3
    __C.TRAIN.WEIGHT_DECAY = 1e-5
    __C.TRAIN.MOMENTUM = 0.90
    __C.TRAIN.NESTEROV = False
    __C.TRAIN.SCHEDULER_MODE = False
    __C.TRAIN.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau')[0]
    __C.TRAIN.SCHEDULER_STEP_MILESTONE = 10
    __C.TRAIN.SCHEDULER_MULTI_MILESTONE = [10]

# ================
# Validation options
# ================
if __C.VALIDATING:
    __C.VALID = edict()

    # Images to use per minibatch
    __C.VALID.BATCH_SIZE = __C.TRAIN.BATCH_SIZE

    # Shuffle the dataset
    __C.VALID.SHUFFLE = False
