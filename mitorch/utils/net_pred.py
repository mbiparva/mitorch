#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch


def cel_prep(p):
    p = p.softmax(dim=1)
    p = p[:, 1, ...]
    p = p.unsqueeze(dim=1)

    return p


def post_proc_pred(p, a, cfg):
    assert isinstance(p, (tuple, list)), 'pack p even if not using Deep_supervision'

    if cfg.AMP and p.dtype is torch.float16:  # dice has one sum that hit inf
        p = [i.to(dtype=torch.float32) for i in p]
        a = a.to(dtype=torch.float32)

    if cfg.MODEL.LOSSES[0]['name'] == 'CrossEntropyLoss':
        p = [cel_prep(i) for i in p]
        a = a.unsqueeze(dim=1).float()

    if cfg.MODEL.LOSSES[0]['with_logits'] or cfg.MODEL.LOSSES[0]['name'] == 'FocalLoss':
        p = [i.sigmoid() for i in p]

    p = torch.mean(torch.stack(p), dim=0)

    return p, a


def pack_pred(p):
    if not isinstance(p, (tuple, list)):
        p = [p]
    return p
