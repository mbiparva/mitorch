#!/usr/bin/env python3

"""Functions that handle saving and loading of checkpoints."""

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import pickle
from collections import OrderedDict
import torch

import utils.logging as logging

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, epoch, best=False):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        epoch (int): the number of epoch for the checkpoint.
        best (bool): best performing so far
    """
    name = "checkpoint_epoch_{:05d}{}.pyth".format(epoch, '_BEST' if best else '')
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = os.listdir(d) if os.path.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        checkpoint_period (int): the frequency of checkpointing.
    """
    return checkpoint_period > 0 and (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_ckpnt, model, optimizer, epoch, cfg, scaler=None, best=False):
    """
    Save a checkpoint.
    Args:
        path_to_ckpnt (str): Path to the checkpoint
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
        scaler (amp scaler): gradient scaler in case AMP is on
        best (bool): is it best performing model?
    """
    # Ensure that the checkpoint dir exists.
    os.makedirs(get_checkpoint_dir(path_to_ckpnt), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.DDP else model.state_dict()
    cfg = cfg.clone()
    if isinstance(cfg.MODEL.SETTINGS, tuple):
        cfg.MODEL.SETTINGS = dict(cfg.MODEL.SETTINGS)[cfg.MODEL.MODEL_NAME]
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_ckpnt, epoch + 1, best=best)
    torch.save(checkpoint, path_to_checkpoint)
    return path_to_checkpoint


def load_checkpoint(
    path_to_checkpoint,
    model,
    distributed_data_parallel=True,
    optimizer=None,
    scaler=None,
):
    """
    Load the checkpoint from the given file.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        distributed_data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        scaler (amp scaler): gradient scaler to use to load the checkpoint
    Returns:
        (int): the number of training epoch of the checkpoint.
    """
    assert os.path.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if distributed_data_parallel else model

    # Load the checkpoint on CPU to avoid GPU mem spike.
    checkpoint = torch.load(path_to_checkpoint, map_location="cpu")
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    if 'scaler' in checkpoint.keys():
        assert scaler is not None, 'gradient scaler are in the checkpoint, turn on cfg.AMP'
        scaler.load_state_dict(checkpoint['scaler'])

    return epoch
