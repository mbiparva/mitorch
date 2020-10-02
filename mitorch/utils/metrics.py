#!/usr/bin/env python3
#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

"""Functions for computing metrics."""

import torch
from netwrapper.functional import dice_coeff, jaccard_index, hausdorff_distance
from sklearn.metrics import f1_score
# Could be implemented manually or called from another external packages like FastAI
# Could add metrics of all different sort of tasks e.g. segmentation, detection, classification


# noinspection PyTypeChecker
def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct_output = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct_output


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def dice_coefficient_metric(p, a, ignore_index):
    # Can use fastai metric too
    return 1 - dice_coeff(
        p,
        a,
        ignore_index=ignore_index,
        reduction='mean'
    ).item()


def jaccard_index_metric(p, a, ignore_index):
    # Can use fastai metric too
    return 1 - jaccard_index(
        p,
        a,
        ignore_index=ignore_index,
        reduction='mean'
    ).item()


def hausdorff_distance_metric(p, a, ignore_index):
    return -hausdorff_distance(
        p.cpu(),  # must be cpu since we get into numpy/scipy scopes
        a.cpu(),
        ignore_index=ignore_index,
        reduction='mean'
    ).item()


def f1_metric(p, a, ignore_index):
    use_ltn = (False, True)[0]  # Mist does not allow to install ltn

    p = p.view(-1)
    a = a.view(-1)

    if use_ltn:
        import pytorch_lightning as ltn
        metric = ltn.metrics.sklearns.F1(average='weighted')
        return metric(
            p,
            a,
        ).item()

    else:
        return f1_score(
            a.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            average='weighted'
        )
