#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn


def pad_if_necessary(x, x_b):
    mode = ('one', 'two')[1]
    size_x = torch.tensor(x.shape[2:], dtype=torch.int)
    size_x_b = torch.tensor(x_b.shape[2:], dtype=torch.int)
    padding_size = size_x - size_x_b
    assert (padding_size >= 0).all(), 'we always pad the backbone outputs not the decoding ones'
    if (padding_size == 0).all():
        return x, x_b
    if mode == 'one':
        padding_size_l = padding_size / 2
        padding_size_r = padding_size - padding_size_l
        padding_tensor = torch.stack((padding_size_l, padding_size_r)).T.flatten().flip(dims=(0,)).tolist()
        return x, nn.functional.pad(x_b, pad=padding_tensor, mode='constant', value=0)
    else:
        return x[:, :, :x_b.size(2), :x_b.size(3), :x_b.size(4)].contiguous(), x_b


def pad_if_necessary_all(x_list, x_b):
    return (
        [
            pad_if_necessary(x, x_b)[0]
            for x in x_list
        ],
        x_b
    )
