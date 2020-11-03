#!/usr/bin/env python3

"""Wrapper to train and test a neural network model."""

#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import _init_lib_path
import os
import argparse
from config.defaults import get_cfg, init_cfg
from test_net import test
from train_net import train as train_single
from train_net_hpo import hpo_main as train_hpo
import torch.distributed as dist


def set_ddp_args(cfg):
    cfg.DDP_CFG.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    cfg.DDP_CFG.RANK = int(os.environ['RANK'])
    cfg.DDP_CFG.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # cfg.DDP_CFG.RANK = int(dist.get_rank())
    # cfg.DDP_CFG.WORLD_SIZE = int(dist.get_world_size())

    cfg.GPU_ID = cfg.DDP_CFG.LOCAL_RANK


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide additional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide medical imaging neural network training and testing pipeline."
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
    cfg = get_cfg(delayed_init=True)
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    if cfg.DDP:
        set_ddp_args(cfg)
        if not cfg.DDP_CFG.RANK:
            cfg = init_cfg(cfg)
        else:
            cfg.OUTPUT_DIR = ''
    else:
        cfg = init_cfg(cfg)

    return cfg


def ddp_init(cfg):
    if cfg.DDP:
        dist.init_process_group(
            backend='nccl', init_method='env://', world_size=cfg.DDP_CFG.WORLD_SIZE, rank=cfg.DDP_CFG.RANK
        )


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)

    # DDP initialization
    ddp_init(cfg)

    # Perform testing.
    if cfg.TEST.ENABLE:
        test(cfg=cfg)

    # Perform training.
    elif cfg.TRAIN.ENABLE:
        if cfg.TRAIN.HPO:
            train_hpo(cfg=cfg)
        else:
            train_single(cfg=cfg)

    else:
        print('no action is performed since both Test and Train are disabled')


if __name__ == "__main__":
    main()
