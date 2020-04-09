import _init_lib_path  # or use export PYTHONPATH=/path/to/lib:$PYTHONPATH
from torch.utils.data import DataLoader
# noinspection PyProtectedMember,PyPep8Naming
from config.defaults import _C as cfg
from data.WMHSegChal import WMHSegmentationChallenge
import data.transforms_mitorch as tf
from torchvision.transforms import Compose


if __name__ == '__main__':
    dataset = WMHSegmentationChallenge(
        cfg,
        Compose([
            tf.ToTensorImageVolume(),
            tf.OrientationToRAI(),
            tf.ResampleTo1mm(),
            tf.ResizeImageVolume(160),
            tf.CenterCropImageVolume(100),
            tf.RandomFlipImageVolume(dim=-1)
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for s in dataloader:
        print(s)
