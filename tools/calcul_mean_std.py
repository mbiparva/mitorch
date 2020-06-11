#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import _init_lib_path  # or use export PYTHONPATH=/path/to/lib:$PYTHONPATH
import torch
from torch.utils.data import DataLoader
# noinspection PyProtectedMember,PyPep8Naming
from config.defaults import _C as cfg
import data.transforms_mitorch as tf
from torchvision.transforms import Compose
from data.build import build_dataset
from tqdm import tqdm
from data.VolSet import collate_fn


def run_calcul(in_dataloader):
    mean, std = list(), list()
    minimum, maximum = list(), list()
    for image, _, _ in tqdm(in_dataloader):
        image = image.permute(1, 0, 2, 3, 4).reshape(image.size(1), -1)
        mean.append(image.mean(1))
        std.append(image.std(1))
        minimum.append(image.min(1)[0])
        maximum.append(image.max(1)[0])

    return (
        torch.stack(mean).mean(0).tolist(),
        torch.stack(std).std(0).tolist(),
        torch.stack(minimum).min(0)[0].tolist(),
        torch.stack(maximum).max(0)[0].tolist(),
    )


def round_tensor(x):
    return [
        round(i, 2) for i in x
    ]


if __name__ == '__main__':
    transforms = Compose([
            tf.ToTensorImageVolume(),
            # tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
    ])
    dataset = build_dataset(
        cfg.TRAIN.DATASET,
        cfg,
        'None',
        transforms
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    mean_ds, std_ds, minimum_ds, maximum_ds = run_calcul(dataloader)

    print(' {:<10}:'.format('mean'), mean_ds, '\n',
          '{:<10}:'.format('std:'), std_ds, '\n',
          '{:<10}:'.format('minimum:'), minimum_ds, '\n',
          '{:<10}:'.format('maximum:'), maximum_ds)
    # print(' {:<10}:'.format('mean'), round_tensor(mean_ds), '\n',
    #       '{:<10}:'.format('std:'), round_tensor(std_ds), '\n',
    #       '{:<10}:'.format('minimum:'), round_tensor(minimum_ds), '\n',
    #       '{:<10}:'.format('maximum:'), round_tensor(maximum_ds))
    print('*** Check dataset for the order of T1 and Flair. (default is this order)')

# WMH-C
# mean      : [255.89, 131.65]
# std:      : [246.27, 91.64]
# minimum:  : [0.0, 0.0]
# maximum:  : [8460.0, 3179.92]

# SRIBIL
# mean      : [326.93, 118.37]
# std:      : [496.88, 126.8]
# minimum:  : [0.0, -66.0]
# maximum:  : [32767.0, 7272.0]

# SRIBILhfb - NORM ON
# mean      : [0.18278566002845764, 0.1672040820121765]
# std:      : [0.018310515210032463, 0.017989424988627434]
# minimum:  : [0.0, 0.0]
# maximum:  : [1.0, 1.0]

# SRIBILhfb - NORM OFF
# mean      : [-0.06902332603931427, -0.0901104062795639]
# std:      : [0.07958264648914337, 0.07952401041984558]
# minimum:  : [-6.7181878089904785, -4.716844081878662]
# maximum:  : [35.84070587158203, 32.10132598876953]
