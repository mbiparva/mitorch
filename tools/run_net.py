#!/usr/bin/env python3

"""Wrapper to train and test a neural network model."""

import _init_lib_path  # or use export PYTHONPATH=/path/to/lib:$PYTHONPATH
import argparse
import sys
import torch

import utils.checkpoint as cu
from config.defaults import get_cfg

from test_net import test
from train_net import train


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide generic neural network training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See lib/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    """
    Given the arguments, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        train(cfg=cfg)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        test(cfg=cfg)


if __name__ == "__main__":
    main()
