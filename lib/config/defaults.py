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

# # Set meters to use for experimental evaluation
# _C.PROJECT.METERS = ['loss', 'label_accuracy']

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
    # _C.PROJECT.DATASET_DIR = '/data2/projects/dataset_hfb'
    _C.PROJECT.DATASET_DIR = '/data3/projects/dataset_hfb'

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

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False


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
_C.TEST.DATASET = _C.TRAIN.DATASET  # ('MNIST', 'FashionMNIST')[0]

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# # -----------------------------------------------------------------------------
# # ResNet options
# # -----------------------------------------------------------------------------
# _C.RESNET = CfgNode()
#
# # Transformation function.
# _C.RESNET.TRANS_FUNC = "bottleneck_transform"
#
# # Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
# _C.RESNET.NUM_GROUPS = 1
#
# # Width of each group (64 -> ResNet; 4 -> ResNeXt).
# _C.RESNET.WIDTH_PER_GROUP = 64
#
# # Apply relu in a inplace manner.
# _C.RESNET.INPLACE_RELU = True
#
# # Apply stride to 1x1 conv.
# _C.RESNET.STRIDE_1X1 = False
#
# #  If true, initialize the gamma of the final BN of each block to zero.
# _C.RESNET.ZERO_INIT_FINAL_BN = False
#
# # Number of weight layers.
# _C.RESNET.DEPTH = 50
#
# # If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# # kernel of 1 for the rest of the blocks.
# _C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]
#
# # Size of stride on different res stages.
# _C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]
#
# # Size of dilation on different res stages.
# _C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]


# # ---------------------------------------------------------------------------- #
# # Batch norm options
# # ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# # BN epsilon.
# _C.BN.EPSILON = 1e-5
#
# # BN momentum.
# _C.BN.MOMENTUM = 0.1
#
# # Precise BN stats.
# _C.BN.USE_PRECISE_STATS = False
#
# # Number of samples use to compute precise bn.
# _C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0


# # -----------------------------------------------------------------------------
# # Nonlocal options
# # -----------------------------------------------------------------------------
# _C.NONLOCAL = CfgNode()
#
# # Index of each stage and block to add nonlocal layers.
# _C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]
#
# # Number of group for nonlocal for each stage.
# _C.NONLOCAL.GROUP = [[1], [1], [1], [1]]
#
# # Instantiation to use for non-local layer.
# _C.NONLOCAL.INSTANTIATION = "dot_product"
#
# # Size of pooling layers used in Non-Local.
# _C.NONLOCAL.POOL = [
#     # Res2
#     [[1, 2, 2], [1, 2, 2]],
#     # Res3
#     [[1, 2, 2], [1, 2, 2]],
#     # Res4
#     [[1, 2, 2], [1, 2, 2]],
#     # Res5
#     [[1, 2, 2], [1, 2, 2]],
# ]


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

# # -----------------------------------------------------------------------------
# # Slowfast options
# # -----------------------------------------------------------------------------
# _C.SLOWFAST = CfgNode()
#
# # Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# # the Slow and Fast pathways.
# _C.SLOWFAST.BETA_INV = 8
#
# # Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# # Fast pathways.
# _C.SLOWFAST.ALPHA = 8
#
# # Ratio of channel dimensions between the Slow and Fast pathways.
# _C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2
#
# # Kernel dimension used for fusing information from Fast pathway to Slow
# # pathway.
# _C.SLOWFAST.FUSION_KERNEL_SZ = 5


# # -----------------------------------------------------------------------------
# # Data options
# # -----------------------------------------------------------------------------
# _C.DATA = CfgNode()
#
# # The path to the data directory.
# _C.DATA.PATH_TO_DATA_DIR = ""
#
# # Video path prefix if any.
# _C.DATA.PATH_PREFIX = ""
#
# # The spatial crop size of the input clip.
# _C.DATA.CROP_SIZE = 224
#
# # The number of frames of the input clip.
# _C.DATA.NUM_FRAMES = 8
#
# # The video sampling rate of the input clip.
# _C.DATA.SAMPLING_RATE = 8
#
# # The mean value of the video raw pixels across the R G B channels.
# _C.DATA.MEAN = [0.45, 0.45, 0.45]
# # List of input frame channel dimensions.
#
# _C.DATA.INPUT_CHANNEL_NUM = [3, 3]
#
# # The std value of the video raw pixels across the R G B channels.
# _C.DATA.STD = [0.225, 0.225, 0.225]
#
# # The spatial augmentation jitter scales for training.
# _C.DATA.TRAIN_JITTER_SCALES = [256, 320]
#
# # The spatial crop size for training.
# _C.DATA.TRAIN_CROP_SIZE = 224
#
# # The spatial crop size for testing.
# _C.DATA.TEST_CROP_SIZE = 256
#
# # Input videos may has different fps, convert it to the target video fps before
# # frame sampling.
# _C.DATA.TARGET_FPS = 30


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 5e-3

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 200

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = False

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

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

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = os.path.join(_C.PROJECT.EXPERIMENT_DIR, _C.TRAIN.DATASET, _C.MODEL.ID)
if not os.path.exists(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 110

# Log period in iters.
_C.LOG_PERIOD = 1  # TODO check to see what this is??????

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 10

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# # ---------------------------------------------------------------------------- #
# # Detection options.
# # ---------------------------------------------------------------------------- #
# _C.DETECTION = CfgNode()
#
# # Whether enable video detection.
# _C.DETECTION.ENABLE = False
#
# # Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
# _C.DETECTION.ALIGNED = True
#
# # Spatial scale factor.
# _C.DETECTION.SPATIAL_SCALE_FACTOR = 16
#
# # RoI transformation resolution.
# _C.DETECTION.ROI_XFORM_RESOLUTION = 7


def _assert_and_infer_cfg(cfg):
    # # RESNET assertions.
    # assert cfg.RESNET.NUM_GROUPS > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP > 0
    # assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
