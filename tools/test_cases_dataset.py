import _init_lib_path  # or use export PYTHONPATH=/path/to/lib:$PYTHONPATH
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
    depth_size = 160
    transformations = ComposePrintSize([
        tf.ToTensorImageVolume(),
        tf.OrientationToRAI(),
        tf.ResampleTo1mm(),
        tf.ResizeImageVolume(max_side, min_side=False),
        tf.PadToSizeVolume(max_side, fill=(-10, 99)),
        # tf.ResizeImageVolume((depth_size, max_side, max_side)),
        tf.RandomFlipImageVolume(dim=-1)
        ])
    dataset = build_dataset(dataset_name, cfg, mode, transformations)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    for cnt, (image, annot, meta) in enumerate(dataloader):
        print('*'*50)
        print(cnt, len(dataloader))
        print(meta)
        print(image.shape)
        print(annot.shape)
        print('\n'*2)


if __name__ == '__main__':
    main()
