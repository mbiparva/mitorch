#!/usr/bin/env python3
#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""Logging."""

import builtins
import decimal
import logging
import sys
import simplejson
import os
import utils.distributed as du


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


# def setup_logging(filepath=None):
#     """
#     Sets up the logging for multiple processes. Only enable the logging for the
#     master process, and suppress logging for the non-master processes.
#     """
#     # Set up logging format.
#     _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
#
#     # Enable logging for the master process.
#     logging.root.handlers = []
#     if filepath is not None:
#         logging.basicConfig(
#             filename=os.path.join(filepath, 'stdout.log'),
#             filemode='w',
#             level=logging.INFO,
#             format=_FORMAT,
#         )
#     else:
#         logging.basicConfig(
#             level=logging.INFO,
#             format=_FORMAT,
#             stream=sys.stdout
#         )


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if du.is_root_proc():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(_FORMAT, datefmt="%m/%d %H:%M:%S")

    if du.is_root_proc():
        if output_dir is not None:
            filename = os.path.join(output_dir, "stdout.log")
            fh = logging.FileHandler(filename)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(plain_formatter)
            logger.addHandler(fh)

        if output_dir is None:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(plain_formatter)
            logger.addHandler(ch)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
