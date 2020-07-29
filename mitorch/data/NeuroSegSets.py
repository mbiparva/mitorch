#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
from .build import DATASET_REGISTRY
import torch
import numpy as np
import torch.utils.data as data
# import SimpleITK as sitk
import nibabel as nib
import re
from torch._six import container_abcs
from abc import ABC, abstractmethod
import data.utils_ext as utils_ext
from collections import defaultdict
import tifffile as tiff
from PIL import Image
import shutil


# noinspection PyUnresolvedReferences
def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, container_abcs.Mapping):  # this is called for the meta
        return batch
    elif isinstance(elem, container_abcs.Sequence):  # this is called at the beginning
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class AutoPatching(ABC, data.Dataset):
    def __init__(self, cfg, mode, transform, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = kwargs.get('num_patches', 1024)
        self.force_image_patching = kwargs.get('force_image_patching', False)
        self.block_size = kwargs.get('block_size', 512)
        self.stride_size = kwargs.get('stride', 64 // 2)  # half of the input size to the networks
        self.image_extensions = ('tif', 'tiff')
        self.sample_path_list = None

        assert self.stride_size < self.block_size

        self._init_pars(cfg, mode, transform)

        if self.force_image_patching:
            self.clean_processed_dir()

        if not self.image_annot_ready():
            self.dirs_dict = self.list_raw_dirs()

            self.load_patch_save()

        self.patch_list = self.list_pro_patches()

    def _init_pars(self, cfg, mode, transform):
        self.cfg = cfg
        self.mode = mode
        self.transform = transform
        self.dataset_path = os.path.join(self.cfg.PROJECT.DATASET_DIR, self.__class__.__name__)

    def list_raw_dirs(self):
        dirs_dict = dict()
        dataset_raw_path = os.path.join(self.dataset_path, 'raw')
        for s in os.listdir(dataset_raw_path):
            if not self.is_data_dir(s):
                continue

            s_path = os.path.join(dataset_raw_path, s)

            # (1) Index Files for all channels
            dirs_dict[s] = self.list_files(s_path)

            # (2) Sanity Check number of files and index numbers match
            self.sanity_check(dirs_dict[s], s_path)

        return dirs_dict

    def list_pro_patches(self):
        patch_list = list()
        dataset_pro_path = os.path.join(self.dataset_path, 'processed')
        for s in os.listdir(dataset_pro_path):
            if not self.is_data_dir(s):
                continue

            s_path = os.path.join(dataset_pro_path, s)

            # (1) Index Files for all channels
            patch_list.extend([
                os.path.join(s_path, i) for i in sorted(os.listdir(s_path))
            ])

        return patch_list

    @abstractmethod
    def list_files(self, s_path):
        raise NotImplementedError

    def load_seq_image_files(self, i_path):
        return sorted([
            i for i in os.listdir(i_path) if i.rpartition('.')[-1] in self.image_extensions
        ])

    @staticmethod
    def sanity_check(sample_files, base_path):
        # number of images
        file_sizes = np.array([len(v) for v in sample_files.values()])
        assert (file_sizes[0] == file_sizes).all(), 'file size in the channel directories do not match'
        keys = list(sample_files.keys())
        # check file ids are the same
        if len(sample_files) > 1:
            for i in range(len(sample_files[keys[0]])):
                file_ids = np.array([v[i].rpartition('_')[-1] for v in sample_files.values()])
                assert (file_ids[0] == file_ids).all(), f'{i}: {file_ids} did not match in ids'
        # check image sizes are the same
        size_list = list()
        for u, v in sample_files.items():
            size_list.extend([
                Image.open(os.path.join(base_path, u, f)).size for f in v
            ])
        size_list = np.vstack(size_list)
        assert (size_list[0] == size_list).all(), f'image sizes in {base_path} do not match'

    @staticmethod
    @abstractmethod
    def load_annot(basename_path):
        raise NotImplementedError

    def load_patch_save(self):
        for u, v in self.dirs_dict.items():
            u_path = os.path.join(self.dataset_path, 'raw', u)
            processed_path = os.path.join(self.dataset_path, 'processed', u)
            if not os.path.exists(processed_path):
                os.mkdir(processed_path)

            annot = self.load_annot(u_path)

            depth, height, width = self.get_volume_size(u_path, v)

            assert annot.shape == (depth, height, width)

            for p in self.grid_generator(depth, height, width):
                # save them in their corresponding directories.
                print(f'{u}: processing the patch @ {p}')

                p_path = self.gen_patch_path(p, processed_path)

                if os.path.exists(p_path):
                    print(f'{p_path} exits and skipped.')
                    continue

                self.load_extract_save_patch(p, v, p_path, u_path, annot)

    @staticmethod
    def get_volume_size(base_path, channel_dicts):
        first_key = list(channel_dicts.keys())[0]
        depth = len(channel_dicts[first_key])
        width, height = Image.open(os.path.join(base_path, first_key, channel_dicts[first_key][0])).size
        return depth, height, width

    def clean_processed_dir(self):
        processed_dir = os.path.join(self.dataset_path, 'processed')
        for f in os.listdir(processed_dir):
            shutil.rmtree(os.path.join(processed_dir, f))

    def image_annot_ready(self):
        return len(os.listdir(os.path.join(self.dataset_path, 'processed'))) > 0

    def grid_generator(self, depth, height, width):
        stride = self.block_size - self.stride_size
        for d in np.arange(0, depth + 1, stride):
            d_start = (depth - self.block_size) if d + self.block_size > depth else d
            for h in np.arange(0, height + 1, stride):
                h_start = (height - self.block_size) if h + self.block_size > height else h
                for w in np.arange(0, width + 1, stride):
                    w_start = (width - self.block_size) if w + self.block_size > width else w
                    yield (
                        d_start,
                        h_start,
                        w_start,
                    )

    @staticmethod
    def gen_patch_path(p, patches_path):
        patch_name = '{}.tiff'.format(
            '_'.join(['{:06}'.format(i) for i in p])
        )

        patch_path = os.path.join(patches_path, patch_name)

        return patch_path

    # noinspection PyTypeChecker
    def load_extract_save_patch(self, p, s_files, patch_path, raw_path, annot):
        d, h, w = p
        crop_box = (
            w,
            h,
            w + self.block_size,
            h + self.block_size,
        )

        s_images = defaultdict(list)
        for u, v in s_files.items():
            for i in v[d: d + self.block_size]:
                i_path = os.path.join(raw_path, u, i)
                s_images[u].append(np.asanyarray(Image.open(i_path).crop(box=crop_box)))

        # merge sheets into one numpy 3D array
        for u, v in s_images.items():
            s_images[u] = np.stack(v)

        annot_cropped = annot[d: d + self.block_size, h: h + self.block_size, w: w + self.block_size]

        s_images = np.stack(list(s_images.values())+[annot_cropped])  # CxZxHxW --- annot is always the last

        s_images = s_images.transpose(1, 0, 2, 3)  # ZxCxHxW

        tiff.imwrite(patch_path, s_images, **{'bigtiff': False, 'imagej': True, 'metadata': {'axes': 'ZCYX'}})

    def __getitem__(self, index):
        p_path = self.patch_list[index]
        p_meta = {
            'sample_path': p_path
        }

        image_tensor, annot_tensor = self.load_image_annot_patch(p_path)

        if self.transform is not None:
            image_tensor_list, annot_tensor_list = list(), list()
            for _ in range(self.cfg.NVT.NUM_MULTI_PATCHES):
                image_tensor, annot_tensor, in_pipe_meta = self.transform((image_tensor, annot_tensor, p_meta))
                image_tensor_list.append(image_tensor)
                annot_tensor_list.append(annot_tensor)

            image_tensor = torch.stack(image_tensor_list)
            annot_tensor = torch.stack(annot_tensor_list)

        return image_tensor, annot_tensor, p_meta

    @staticmethod
    def annot_sanity_check(annot_tensor):
        annot_unq = torch.unique(annot_tensor)
        if not all([i in (0, 1) for i in annot_unq]):
            assert len(annot_unq) == 2 and (annot_unq == torch.tensor((0, 255))).all(), f'{annot_unq} is not in range'
            return annot_tensor / 255
        return annot_tensor

    def load_image_annot_patch(self, p_path):
        patch_array = tiff.imread(p_path).astype(np.float32)

        patch_tensor = torch.tensor(
                data=patch_array,
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            )

        image_tensor = patch_tensor[:, :-1, :, :]
        annot_tensor = patch_tensor[:, -1, :, :]  # the last channels is always the annotation

        image_tensor = image_tensor.permute((1, 0, 2, 3))
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        annot_tensor = self.annot_sanity_check(annot_tensor)

        return image_tensor, annot_tensor

    def __len__(self):
        return len(self.patch_list)

    @abstractmethod
    def is_data_dir(self, sample_dir_name):
        pass


@DATASET_REGISTRY.register()
class TRAP(AutoPatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_data_dir(self, sample_dir_name):
        return sample_dir_name[0].isdigit()

    def list_files(self, s_path):
        output_file_list = dict()
        for i in ('Ex_1_Em_1_destriped_stitched', 'Ex_0_Em_0_destriped_stitched'):
            i_path = os.path.join(s_path, i)
            output_file_list[i] = self.load_seq_image_files(i_path)

        return output_file_list

    @staticmethod
    def load_annot(basename_path):
        annot_file = os.path.join(basename_path, 'seg_bin_cfos.tif')
        return tiff.imread(annot_file)


@DATASET_REGISTRY.register()
class CAPTURE(AutoPatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_data_dir(self, sample_dir_name):
        return re.match(r'^[a-zA-Z]\d+$', sample_dir_name) is not None

    def list_files(self, s_path):
        output_file_list = dict()
        for i in ('green', ):
            i_path = os.path.join(s_path, i)
            output_file_list[i] = self.load_seq_image_files(i_path)

        return output_file_list

    @staticmethod
    def load_annot(basename_path):
        annot_file = os.path.join(basename_path, 'segmentation_green_virus', 'seg_bin_virus.tif')
        return tiff.imread(annot_file)