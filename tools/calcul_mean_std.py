#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
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
from data.VolSet import collate_fn as collate_fn_vol
from data.NeuroSegSets import collate_fn as collate_fn_pat


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
        tf.RandomCropImageVolumeConditional(cfg.DATA.CROP_SIZE, prand=True,
                                            num_attemps=cfg.NVT.RANDOM_CROP_NUM_ATTEMPS,
                                            threshold=cfg.NVT.RANDOM_CROP_THRESHOLD),
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
        collate_fn=collate_fn_pat,
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
    # print('*** Check dataset for the order of T1 and Flair. (default is this order)')

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

# MAGED PREPROCESSED
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

# NON-PREPROCESSED
# SRIBILhfb - NORM ON
# mean      : [0.058173052966594696, 0.044205766171216965, 0.04969067499041557]  # T1, FL, T2
# std:      : [0.021794982254505157, 0.02334374189376831, 0.024663571268320084]
# minimum:  : [0.0, 0.0, 0.0]
# maximum:  : [1.0, 1.0, 1.0]

# SRIBILhfb - NORM OFF
# mean      : [381.2473449707031, 115.29212188720703, 154.54150390625]
# std:      : [644.0137939453125, 150.35133361816406, 181.5745849609375]
# minimum:  : [0.0, 0.0, 0.0]
# maximum:  : [32767.0, 9614.0, 13264.0]


# NON-PREPROCESSED
# SRIBIL - NORM ON
#  mean      : [0.05927010998129845, 0.052446719259023666]
#  std:      : [0.03302580118179321, 0.02068150043487549]
#  minimum:  : [0.0, 0.0]
#  maximum:  : [1.0, 1.0]

# NVT
# with MinMax, 16000, 16 patches
#  mean      : [0.028548698872327805, 0.12191379070281982]
#  std:      : [0.03772956132888794, 0.061269789934158325]

# no selection
#  mean      : [0.20209592580795288, 0.28169071674346924]
#  std:      : [0.017290744930505753, 0.06943022459745407]

# without MinMax, 16000, 16 patches
#  mean      : [256.42889404296875, 380.6856689453125]
#  std:      : [64.1461410522461, 78.29484558105469]


