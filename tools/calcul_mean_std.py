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
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    mean_ds, std_ds, minimum_ds, maximum_ds = run_calcul(dataloader)

    print(' {:<10}:'.format('mean'), round_tensor(mean_ds), '\n',
          '{:<10}:'.format('std:'), round_tensor(std_ds), '\n',
          '{:<10}:'.format('minimum:'), round_tensor(minimum_ds), '\n',
          '{:<10}:'.format('maximum:'), round_tensor(maximum_ds))
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
