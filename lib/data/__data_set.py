from utils.config import cfg

import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
from data.temporal_sampling import TemporalSampler
import cv2
import numpy as np
import random
import data.spatial_transformation as tf
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

_NUM_OPENCV_THREADS = 8
_CLIP_FLOW = True
_CLIP_L, _CLIP_U = -20, 20
cv2.setNumThreads(_NUM_OPENCV_THREADS)


class LCTDataset(Dataset):
    def __init__(self, spatial_trans):
        self.spatial_trans = spatial_trans
        self.dataset_path = cfg.DATASET_ROOT

        self.temporal_sampler = TemporalSampler(cfg.FRAME_SAMPLING_METHOD)

        self.clips = list()
        for f in ('ROIs_lc0', 'ROIs_lc3', 'ROIs_lc4'):
            self.clips = self.clips + LCTDataset.file_indexer(
                os.path.join(self.dataset_path, f, cfg.CROP_MODE)
            )

        self.labels = ('0', '3', '4')

    @staticmethod
    def file_indexer(dataset_path):
        # list all files
        f_list = list()
        for l in sorted(os.listdir(dataset_path)):
            f_name = l.split('.')[0]
            f_list.append(f_name.split('_'))

        # group them in a hash map by the id
        f_dict = defaultdict(list)
        for f in f_list:
            f_dict[f[7]].append(f)

        # split ids with separate sequences
        f_unfold = list()
        for k, (u, v) in enumerate(f_dict.items()):
            assert len(v)
            print(k, len(f_unfold))
            n_pre = None
            n_start = 0
            for i, j in enumerate(v):
                added = False
                n = int(j[5])
                if n_pre is None:
                    n_pre = n
                else:
                    if n == n_pre + 1:
                        n_pre = n
                    else:
                        f_unfold.append(v[n_start:i])
                        n_pre = n
                        n_start = i
                        print('split ---')
                        assert len(f_unfold[-1])
                        print('++++++', len(f_unfold) - 1, len(f_unfold[-1]))
                        added = True
                print(n)
            if not added:
                f_unfold.append(v[n_start:i + 1])
                assert len(f_unfold[-1])
                print('++++++', len(f_unfold) - 1, len(f_unfold[-1]))
                added = True

            assert added
            print(u, 'done')

        def get_size(in_clip):
            def get_frame_int(in_frame):
                return int(in_frame[5])

            start, end = in_clip[0], in_clip[-1]
            start, end = get_frame_int(start), get_frame_int(end)
            return end - start + 1

        # some sanity check
        clip_size = []
        for k, i in enumerate(f_unfold):
            print(k, len(i), 'Passed')
            assert len(i)  # len must be non-zero
            assert len(i) == get_size(i)  # len must be equal to the frame number range
            clip_size.append(len(i))
            label_check = True
            for j in i:
                label_check = label_check and j[-1] == i[0][-1]
            assert label_check  # all frames must have the same class label

        # final iteration; filter out small clips
        min_clip_size = 10
        clips = []
        for i in f_unfold:
            if len(i) > min_clip_size:
                clips.append(i)

        print('--- done ---', os.path.basename(os.path.dirname(dataset_path)), '|', os.path.basename(dataset_path))
        return clips

    def __getitem__(self, index):
        uv = ['u', 'v']
        f_names = self.clips[index]
        f_label = self.labels.index(f_names[0][-1])
        f_frames = len(f_names)
        f_name = 'img_{}_{}_{}_{}_{}.png'.format(*f_names[0])

        f_frame_list = self.temporal_sampler.frame_sampler(f_frames)

        frame_bank = self.load_frames(f_names, f_frame_list)

        # extract flows
        flow_bank = self.extract_flow(frame_bank)

        self.spatial_trans.randomize_parameters()
        if cfg.FRAME_RANDOMIZATION:
            self.spatial_trans.randomize_parameters()
        frame_bank_t = [self.spatial_trans(i, 'image') for i in frame_bank]
        flow_bank_t = [
            self.spatial_trans(i, 'flow_{}'.format(uv[k % 2])) for k, i in enumerate(flow_bank)
        ]

        frames_p, flows_p = self.pack_frames(frame_bank_t, flow_bank_t)

        return frames_p, flows_p, {'file_name': f_name, 'nframes': f_frames, 'label': f_label}

    def __len__(self):
        return len(self.clips)

    def load_frames(self, fnames, flist):
        frame_bank = []

        for f in flist:
            fname = '{}.png'.format('_'.join(fnames[int(f)]))
            frame_bank.append(Image.open(
                os.path.join(self.dataset_path, 'ROIs_lc{}'.format(fnames[int(f)][-1]), cfg.CROP_MODE, fname)
            ))

        return frame_bank

    @staticmethod
    def pack_frames(frames, flows):
        frames = frames[2::4]
        frames_o = torch.stack(frames).transpose(1, 0)
        flows = torch.cat(flows)
        flows_o = flows.view(5, 8, flows.size(1), flows.size(2)).transpose(1, 0)
        return frames_o, flows_o

    @staticmethod
    def preprocess_frame(video_frame):
        video_frame_gray = video_frame.convert('L')
        video_frame_gray = np.asarray(video_frame_gray)
        return video_frame_gray.astype(np.float32) / 255

    @staticmethod
    def extract_flow(video):
        flow_bank = []
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        f_prev = LCTDataset.preprocess_frame(video[0])

        for v in video[1:]:
            f_next = LCTDataset.preprocess_frame(v)
            of = tvl1.calc(f_prev, f_next, None)
            if _CLIP_FLOW:
                of = of.clip(min=_CLIP_L, max=_CLIP_U) - _CLIP_L
                of = ((of / (_CLIP_U - _CLIP_L)) * 255.).astype(np.uint8)

            flow_bank.append(Image.fromarray(of[..., 0]))  # u
            flow_bank.append(Image.fromarray(of[..., 1]))  # v

        flow_bank.append(Image.fromarray(of[..., 0]))  # u
        flow_bank.append(Image.fromarray(of[..., 1]))  # v

        return flow_bank


class LCTDatasetSave(LCTDataset):
    def __init__(self, spatial_trans):
        super().__init__(spatial_trans)
        self.resize_func = tf.Resize(cfg.SPATIAL_INPUT_SIZE[0])

    def __getitem__(self, index):
        import gc
        gc.collect()  # gc collect to collect open files and variables

        f_names = self.clips[index]
        f_label = self.labels.index(f_names[0][-1])
        f_frames = len(f_names)
        f_name = '{}.png'.format('_'.join(f_names[0]))

        # f_frame_list = self.temporal_sampler.frame_sampler(f_frames)
        # assert len(f_frame_list) == cfg.NFRAMES_PER_VIDEO

        f_frame_list = list(range(f_frames))

        frame_bank = self.load_frames(f_names, f_frame_list)

        # the resize all of the frames to a fixed size, for the sake of OF Function
        frame_bank = [self.resize_func(i) for i in frame_bank]

        flow_bank = self.extract_flow(frame_bank)

        self.save_frame_flow(index, f_names, f_frame_list, frame_bank, flow_bank)

        return {'file_name': f_name, 'nframes': f_frames, 'label': f_label}

    def save_frames(self, ids, fnames, flist, frames):
        dir_path = os.path.normpath(os.path.join(self.dataset_path, 'rgb_of_processed', cfg.CROP_MODE, 'rgb'))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i, f in enumerate(flist):
            fname = '{}_{}.png'.format(ids, '_'.join(fnames[int(f)]))
            frames[i].save(os.path.join(dir_path, fname))

    def save_flow(self, ids, fnames, flist, flows):
        dir_path_ofu = os.path.normpath(os.path.join(self.dataset_path, 'rgb_of_processed', cfg.CROP_MODE, 'ofu'))
        dir_path_ofv = os.path.normpath(os.path.join(self.dataset_path, 'rgb_of_processed', cfg.CROP_MODE, 'ofv'))
        if not os.path.exists(dir_path_ofu):
            os.makedirs(dir_path_ofu)
        if not os.path.exists(dir_path_ofv):
            os.makedirs(dir_path_ofv)
        for i, f in enumerate(flist):
            fname = '{}_{}.png'.format(ids, '_'.join(fnames[int(f)]))
            flows[i*2].save(os.path.join(dir_path_ofu, fname))
            flows[i*2+1].save(os.path.join(dir_path_ofv, fname))

    def save_frame_flow(self, ids, fnames, flist, frames, flows):
        self.save_frames(ids, fnames, flist, frames)
        self.save_flow(ids, fnames, flist, flows)


class LCTDatasetLight(LCTDataset):
    def __init__(self, spatial_trans):
        self.dataset_path = os.path.normpath(os.path.join(cfg.DATASET_ROOT, 'rgb_of_processed', cfg.CROP_MODE))
        self.dataset_path_rgb = os.path.normpath(os.path.join(self.dataset_path, 'rgb'))
        self.dataset_path_ofu = os.path.normpath(os.path.join(self.dataset_path, 'ofu'))
        self.dataset_path_ofv = os.path.normpath(os.path.join(self.dataset_path, 'ofv'))

        self.temporal_sampler = TemporalSampler(cfg.FRAME_SAMPLING_METHOD)
        self.spatial_trans = spatial_trans

        # list all files
        f_dict = defaultdict(list)
        for l in sorted(os.listdir(self.dataset_path_rgb)):
            f_ind = l.split('_')[0]
            f_dict[int(f_ind)].append(l)

        # skip the last 10 frames after t+TTE
        self.f_dict_filt = defaultdict(list)
        buffer = 5
        for u, v in f_dict.items():
            if len(v) - cfg.EFOFFSET - cfg.tTTE_OFFSET - buffer <= 0:  # -5 is just a buffer
                continue
            if cfg.EFOFFSET > 0:
                self.f_dict_filt[u] = v[:-cfg.EFOFFSET]
            self.f_dict_filt[u] = self.f_dict_filt[u][
                -cfg.tTTE-cfg.tTTE_OFFSET:-cfg.tTTE_OFFSET
            ] if cfg.tTTE_OFFSET > 0 else self.f_dict_filt[u][-cfg.tTTE:]
            # check for those indices that have nframes < cfg.NFRAMES_PER_VIDEO
            if len(self.f_dict_filt[u]) < cfg.NFRAMES_PER_VIDEO:
                self.f_dict_filt[u] = random.choices(self.f_dict_filt[u], k=cfg.NFRAMES_PER_VIDEO)
                self.f_dict_filt[u] = sorted(self.f_dict_filt[u])

        self.labels = ('0', '3', '4')
        self.f_dict_keys = list(self.f_dict_filt)

    def __len__(self):
        return len(self.f_dict_keys)

    def __getitem__(self, index):
        uv = ['u', 'v']
        f_key = self.f_dict_keys[index]
        f_names = self.f_dict_filt[f_key]
        f_name = f_names[0]
        f_label = self.labels.index(f_name.split('_')[-1].split('.')[0])
        f_frames = len(f_names)

        # resample frames when tTTE > NFRAMES_PER_VIDEO
        if f_frames > cfg.NFRAMES_PER_VIDEO:
            f_names = [f_names[i] for i in self.temporal_sampler.frame_sampler(f_frames)]

        # load frames
        frame_bank = self.load_frames(f_names)

        # load flows
        flow_bank = self.load_flows(f_names)

        self.spatial_trans.randomize_parameters()
        if cfg.FRAME_RANDOMIZATION:
            self.spatial_trans.randomize_parameters()
        frame_bank_t = [self.spatial_trans(i, 'image') for i in frame_bank]
        flow_bank_t = [
            self.spatial_trans(i, 'flow_{}'.format(uv[k % 2])) for k, i in enumerate(flow_bank)
        ]

        frames_p, flows_p = self.pack_frames(frame_bank_t, flow_bank_t)

        return frames_p, flows_p, {'file_name': f_name, 'nframes': f_frames, 'label': f_label}

    def load_frames(self, fnames):
        frame_bank = []

        for f in fnames:
            frame_bank.append(Image.open(os.path.join(self.dataset_path_rgb, f)))

        return frame_bank

    def load_flows(self, fnames):
        flow_bank = []

        for f in fnames:
            flow_bank.append(Image.open(os.path.join(self.dataset_path_ofu, f)))
            flow_bank.append(Image.open(os.path.join(self.dataset_path_ofv, f)))

        return flow_bank

    @staticmethod
    def pack_frames(frames, flows):
        fps = 5
        stride = cfg.NFRAMES_PER_VIDEO // fps
        frames = frames[stride//2::stride]  # offset with half of stride
        frames_o = torch.stack(frames).transpose(1, 0)
        flows = torch.cat(flows)
        flows_o = flows.view(fps, len(flows)//fps, flows.size(1), flows.size(2)).transpose(1, 0)
        return frames_o, flows_o
