#!/usr/bin/env python3

#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

import os
import time
import socket
import pprint
import numpy as np

from trainer import Trainer
from evaluator import Evaluator
import utils.logging as logging
import utils.misc as misc
import utils.checkpoint as checkops

import torch
from torch.utils.tensorboard import SummaryWriter
from netwrapper.net_wrapper import NetWrapper

logger = logging.get_logger(__name__)


class EpochLoop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainer, self.evaluator = None, None
        self.device, self.net_wrapper = None, None
        self.tb_logger_writer = None

        self.setup_gpu()

    def setup_gpu(self):
        cuda_device_id = self.cfg.GPU_ID
        if self.cfg.NUM_GPUS > 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(cuda_device_id))
        else:
            self.device = torch.device('cpu')

    def setup_tb_logger(self):
        self.tb_logger_writer = SummaryWriter(self.cfg.OUTPUT_DIR)
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'cfg.yml'), 'w') as outfile:
            self.cfg.dump(stream=outfile)

    def tb_logger_update(self, e, worker):
        if e == 0 and self.tb_logger_writer is None:
            self.setup_tb_logger()
        worker.tb_logger_update(self.tb_logger_writer, e)

    def save_checkpoint(self, cur_epoch):
        # TODO add if it is the best, save it separately too
        if checkops.is_checkpoint_epoch(cur_epoch, self.cfg.TRAIN.CHECKPOINT_PERIOD):
            self.net_wrapper.save_checkpoint(self.cfg.OUTPUT_DIR, cur_epoch)

    def load_checkpoint(self):
        if self.cfg.TRAIN.AUTO_RESUME and checkops.has_checkpoint(self.cfg.OUTPUT_DIR):
            logger.info("Load from last checkpoint.")
            last_checkpoint = checkops.get_last_checkpoint(self.cfg.OUTPUT_DIR)
            checkpoint_epoch = self.net_wrapper.load_checkpoint(last_checkpoint)
            start_epoch = checkpoint_epoch + 1
        elif self.cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
            logger.info("Load from given checkpoint file.")
            checkpoint_epoch = self.net_wrapper.load_checkpoint(self.cfg.TRAIN.CHECKPOINT_FILE_PATH)
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0

        return start_epoch

    def check_if_validating(self, cur_epoch):
        if misc.is_eval_epoch(self.cfg, cur_epoch):
            self.evaluator_epoch_loop(cur_epoch)
            logger.info('*** Done validating at epoch {}'.format(cur_epoch))
            return self.evaluator.meters.get_epoch_loss()

        return None

    def lr_scheduling(self, eval_loss_avg_last):
        self.net_wrapper.schedule_step(eval_loss_avg_last)

    def trainer_epoch_loop(self, start_epoch):
        for cur_epoch in range(start_epoch, self.cfg.SOLVER.MAX_EPOCH):
            self.trainer.set_net_mode(self.net_wrapper.net_core)

            self.trainer.meters.reset()

            self.trainer.batch_loop(self.net_wrapper, cur_epoch)

            self.trainer.meters.log_epoch_stats(cur_epoch, 'train')

            self.save_checkpoint(cur_epoch)

            self.tb_logger_update(cur_epoch, self.trainer)

            eval_loss_avg_last = self.check_if_validating(cur_epoch)

            self.lr_scheduling(eval_loss_avg_last)

    def evaluator_epoch_loop(self, start_epoch):
        self.evaluator.set_net_mode(self.net_wrapper.net_core)

        self.evaluator.meters.reset()

        self.evaluator.batch_loop(self.net_wrapper, start_epoch)

        # Log epoch stats. For the moment, valid and test are the same. For multi-view, multi-crops, modify it later.
        self.evaluator.meters.log_epoch_stats(start_epoch, 'valid')

        self.tb_logger_update(start_epoch, self.evaluator)

    def main_setup(self):
        np.random.seed(self.cfg.RNG_SEED)
        torch.manual_seed(self.cfg.RNG_SEED)

        socket_name = socket.gethostname()
        logging.setup_logging(
            filepath=self.cfg.OUTPUT_DIR if 'scinet' in socket_name or 'computecanada' in socket_name else None
        )

        logger.info("Train with config:")
        logger.info(pprint.pformat(self.cfg))

    def create_sets(self):
        self.trainer = Trainer(self.cfg, self.device) if self.cfg.TRAIN.ENABLE else None
        self.evaluator = Evaluator(self.cfg, self.device) if self.cfg.VALID.ENABLE else None

    def setup_net(self):
        self.net_wrapper = NetWrapper(self.device, self.cfg)

    def run(self, start_epoch):
        logger.info("Start epoch: {}".format(start_epoch + 1))

        if self.cfg.TRAIN.ENABLE:
            self.trainer_epoch_loop(start_epoch)
        elif self.cfg.VALID.ENABLE:
            self.evaluator_epoch_loop(0)
        elif self.cfg.TESTING:  # Test mode is only different from valid once data is loaded
            raise NotImplementedError('TESTING mode is not implemented yet')
        else:
            raise NotImplementedError('One of {TRAINING, VALIDATING, TESTING} must be set to True')

        self.tb_logger_writer.close()

    def main(self):
        self.main_setup()

        self.create_sets()

        self.setup_net()

        start_epoch = self.load_checkpoint()

        self.run(start_epoch)
