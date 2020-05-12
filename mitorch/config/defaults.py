#!/usr/bin/env python3

"""Configs."""
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

# Root directory of project
_C.PROJECT.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory
_C.PROJECT.DATASET_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'dataset'))

# Model directory
_C.PROJECT.MODELS_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'lib', 'models'))

# Experiment directory
_C.PROJECT.EXPERIMENT_DIR = os.path.abspath(os.path.join(_C.PROJECT.ROOT_DIR, 'experiments'))

# Training, Validation, and Test Split Ratio
_C.PROJECT.TVSR = 0.80
_C.PROJECT.TSR = 0.0

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = ('WMHSegmentationChallenge', 'SRIBIL', 'SRIBILhfb')[2]

# Choose to load hfb annotations
_C.TRAIN.SRIBIL_HFB_ANNOT = True

if socket.gethostname() == 'cerveau.sri.utoronto.ca':
    _C.PROJECT.DATASET_DIR = '/data2/projects/dataset_hfb'

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1

# Shuffle the training data, valid / test are never shuffled
_C.TRAIN.SHUFFLE = True

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs. < 1 will deactivate
_C.TRAIN.CHECKPOINT_PERIOD = 20

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
_C.TEST.DATASET = _C.TRAIN.DATASET

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""


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

_C.MODEL.ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

# Model architecture.
_C.MODEL.ARCH = "unet3d"

# Model name
_C.MODEL.MODEL_NAME = ('Unet3D', )[0]

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 1

# Loss function.
_C.MODEL.LOSS_FUNC = ('CrossEntropyLoss', 'DiceLoss')[1]

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
_C.MODEL.INPUT_CHANNELS = 2  # T1 & FLAIR


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The spatial max side size of the input volume.
_C.DATA.MAX_SIDE_SIZE = 192

# The spatial crop size of the input volume.
_C.DATA.CROP_SIZE = 160

# The spatial crop scale of the input volume.
_C.DATA.CROP_SCALE = (0.7, 1.0)

# The mean value of the volume raw voxels across the T1 and Flair channels.
_C.DATA.MEAN = [0.18278566002845764, 0.1672040820121765]  # TODO add it to the init of dataset to automatically fine it.

# The standard deviation value of the volume raw voxels across the T1 and Flair channels.
_C.DATA.STD = [0.018310515210032463, 0.017989424988627434]

_C.DATA.PADDING_MODE = ('mean', 'median', 'min', 'max')[0]


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 5e-3

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 200

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Nesterov momentum.
_C.SOLVER.NESTEROV = False

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# TODO add warup and scheduler hypers here

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = ('sgd', 'adam')[1]

# Enable Scheduler
_C.SOLVER.SCHEDULER_MODE = False

# Set the type of scheduler
_C.SOLVER.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau', 'cosine')[0]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing). 0 Means don't use GPU.
_C.NUM_GPUS = 1

# Default GPU device id
_C.GPU_ID = 0

# Output basedir.
_C.OUTPUT_DIR = os.path.join(_C.PROJECT.EXPERIMENT_DIR, _C.TRAIN.DATASET, _C.MODEL.ID)
if not os.path.exists(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)

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


def _assert_and_infer_cfg(cfg):
    # TODO add assertion respectively
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())