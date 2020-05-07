import _init_lib_path  # or use export PYTHONPATH=/path/to/lib:$PYTHONPATH
import time
from torch.utils.data import DataLoader
# noinspection PyProtectedMember,PyPep8Naming
from config.defaults import _C as cfg
import data.transforms_mitorch as tf
from torchvision.transforms import Compose
from data.build import build_dataset
from data.VolSet import collate_fn


class ComposePrintSize(Compose):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, volume):
        # print('\n\n', '-'*50)
        # print(volume[2])
        # print('-'*50)
        for t in self.transforms:
            # if t.__class__.__name__ == 'PadToSizeVolume':
            # if t.__class__.__name__ == 'RandomCropImageVolume':
            # if t.__class__.__name__ == 'RandomResizedCropImageVolume':
            # if t.__class__.__name__ == 'CenterCropImageVolume':
            #     print('we are in ...')
            volume = t(volume)
            # image, annot, meta = volume
            # print("image: {image}, annot: {annot}".format(
            #     image=list(image.shape), annot=list(annot.shape)
            # ))
        return volume


def main():
    dataset_name = cfg.TRAIN.DATASET
    mode = 'training'
    max_side = 256
    crop_size, crop_scale = 224, (0.70, 0.90)
    transformations = ComposePrintSize([
        tf.ToTensorImageVolume(),
        tf.OrientationToRAI(),
        tf.ResampleTo1mm(),
        tf.ResizeImageVolume(max_side, min_side=False),
        tf.PadToSizeVolume(max_side, padding_mode=('mean', 'median', 'min', 'max')[0]),
        tf.CenterCropImageVolume(crop_size),
        # tf.RandomCropImageVolume(224),
        # tf.RandomResizedCropImageVolume(crop_size, scale=crop_scale),
        tf.RandomFlipImageVolume(dim=-1),
        tf.NormalizeMinMaxVolume(max_div=True, inplace=True),
        tf.NormalizeMeanStdVolume(
            mean=[0.18278566002845764, 0.1672040820121765],
            std=[0.018310515210032463, 0.017989424988627434],
            inplace=True
        ),
        ])
    dataset = build_dataset(dataset_name, cfg, mode, transformations)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    start = time.time()
    for cnt, (image, annot, meta) in enumerate(dataloader):
        print('*'*50)
        print(cnt, len(dataloader))
        print(meta)
        print(image.shape)
        print(annot.shape)
        print('\n'*2)
    duration = time.time() - start
    h, duration = divmod(duration, 3600)
    m, s = divmod(duration, 60)
    print('took: {} h: {} m: {} s'.format(h, m, s))


if __name__ == '__main__':
    main()
