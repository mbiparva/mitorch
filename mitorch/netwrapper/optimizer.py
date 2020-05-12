#!/usr/bin/env python3
#  Copyright (c) 2020.
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institure (SRI)

"""Optimizer."""

import torch
import torch.optim as optim


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


# noinspection PyTypeChecker
def construct_scheduler(optimizer, cfg):
    if cfg.SOLVER.SCHEDULER_MODE:
        if cfg.SOLVER.SCHEDULER_TYPE == 'step':
            return optim.lr_scheduler.StepLR(optimizer,
                                             step_size=10,
                                             gamma=0.1)
        elif cfg.SOLVER.SCHEDULER_TYPE == 'step_restart':
            return StepLRestart(optimizer,
                                step_size=4,
                                restart_size=8,
                                gamma=0.1)
        elif cfg.SOLVER.SCHEDULER_TYPE == 'multi':
            return optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[10, 15, 25],
                                                  gamma=0.1)
        elif cfg.SOLVER.SCHEDULER_TYPE == 'lambda':
            def lr_lambda(e):
                return 1 if e < 5 else .5 if e < 10 else .1 if e < 15 else .01

            return optim.lr_scheduler.LambdaLR(optimizer,
                                               lr_lambda=lr_lambda)
        elif cfg.SOLVER.SCHEDULER_TYPE == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.1,
                                                        patience=10,
                                                        cooldown=0,
                                                        verbose=True)
        elif cfg.SOLVER.SCHEDULER_TYPE == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=cfg.SOLVER.MAX_EPOCH,
                                                        eta_min=1e-8)
        else:
            raise NotImplementedError


# noinspection PyProtectedMember
class StepLRestart(optim.lr_scheduler._LRScheduler):
    """The same as StepLR, but this one has restart.
    """
    def __init__(self, optimizer, step_size, restart_size, gamma=0.1, last_epoch=-1):
        self.base_lrs, self.last_epoch = None, None
        self.step_size = step_size
        self.restart_size = restart_size
        assert self.restart_size > self.step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** ((self.last_epoch % self.restart_size) // self.step_size)
                for base_lr in self.base_lrs]
