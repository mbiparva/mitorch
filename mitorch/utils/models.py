#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import torch
import torch.nn as nn


def pad_if_necessary(x, x_b):
    mode = ('one', 'two')[1]
    size_x = torch.tensor(x.shape[2:], dtype=torch.int)
    size_x_b = torch.tensor(x_b.shape[2:], dtype=torch.int)
    padding_size = size_x - size_x_b
    if (padding_size == 0).all():
        return x, x_b

    if (padding_size >= 0).all():
        if mode == 'one':
            padding_size_l = padding_size / 2
            padding_size_r = padding_size - padding_size_l
            padding_tensor = torch.stack((padding_size_l, padding_size_r)).T.flatten().flip(dims=(0,)).tolist()
            return x, nn.functional.pad(x_b, pad=padding_tensor, mode='constant', value=0)
        else:
            return x[:, :, :x_b.size(2), :x_b.size(3), :x_b.size(4)].contiguous(), x_b
    else:
        if mode == 'one':
            padding_size_l = padding_size / 2
            padding_size_r = padding_size - padding_size_l
            padding_tensor = torch.stack((padding_size_l, padding_size_r)).T.flatten().flip(dims=(0,)).tolist()
            x, x_b = nn.functional.pad(x, pad=padding_tensor, mode='constant', value=0), x_b
        else:
            x, x_b = x, x_b[:, :, :x.size(2), :x.size(3), :x.size(4)].contiguous()

        return pad_if_necessary(x, x_b)


def pad_if_necessary_all(x_list, x_b):
    for i, x in enumerate(x_list):
        x_list[i], x_b = pad_if_necessary(x, x_b)

    return x_list, x_b


def is_3d(size, is_3d_flag, lb=1):
    if isinstance(size, (tuple, list)):
        assert len(size) == 3, 'expects 3D iterables'
        return (
            is_3d(size[0], is_3d_flag, lb=lb),
            size[1],
            size[2],
        )
    elif isinstance(size, (int, float)):
        return size if is_3d_flag else lb
    else:
        raise NotImplementedError
