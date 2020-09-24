#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import torch
import numpy as np
from data.SRIBILSet import SRIBILBase


class TestSet(SRIBILBase):
    def __init__(self, cfg, mode, transform, prefix_name):
        super().__init__(cfg, mode, transform)
        self.prefix_name = prefix_name

    def index_samples(self):
        return [self.cfg.TEST.DATA_PATH] if not self.cfg.TEST.BATCH_MODE else [
            os.path.join(self.cfg.TEST.DATA_PATH, i)
            for i in sorted(os.listdir(self.cfg.TEST.DATA_PATH))
            ]

    def get_data_tensor(self, in_pipe_data):
        in_pipe_data = {
            u: torch.tensor(
                data=v,
                dtype=torch.float,
                device='cpu',
                requires_grad=False
            )
            for u, v in in_pipe_data.items()
        }

        if 'annot' in in_pipe_data:
            annot_tensor = in_pipe_data.pop('annot')
        else:
            annot_tensor = torch.zeros_like(list(in_pipe_data.values())[0])

        image_tensor = list(in_pipe_data.values())

        annot_tensor = self.curate_annotation(annot_tensor, ignore_index=self.cfg.MODEL.IGNORE_INDEX)

        return image_tensor, annot_tensor

    # The reason we override run_sanity_checks & getitem here in PPMI is that there is a
    # misalignment between Flair and other modalities.
    # Flair has shape (x, x, x) others have (x, x, x, 1) which is wrong
    @staticmethod
    def run_sanity_checks(in_pipe_meta):
        return in_pipe_meta['fl']  # fl is correct, we skip sanity check since they don't match

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]

        in_pipe_data = self.find_data_files_path(sample_path)
        in_pipe_data = self.load_data(in_pipe_data,
                                      enforce_nib_canonical=self.cfg.DATA.ENFORCE_NIB_CANONICAL,
                                      enforce_diag=self.cfg.DATA.ENFORCE_DIAG,
                                      dtype=np.float32)
        in_pipe_data, in_pipe_meta = self.extract_data_meta(in_pipe_data)
        for u, v in in_pipe_data.items():
            if v.ndim == 4:
                assert v.shape[-1] == 1
                in_pipe_data[u] = in_pipe_data[u].squeeze(-1)

        try:
            in_pipe_meta = self.run_sanity_checks(in_pipe_meta)

            in_pipe_meta['sample_path'] = sample_path

            image_tensor, annot_tensor = self.get_data_tensor(in_pipe_data)
        except Exception as e:
            print('exception {} raised in the dataloader getitem function for {}'.format(e, sample_path))
            raise e

        image_tensor = torch.stack(image_tensor, dim=-1)  # D x H x W x C
        annot_tensor = annot_tensor.unsqueeze(dim=0)

        if self.transform is not None:
            image_tensor, annot_tensor, in_pipe_meta = self.transform((image_tensor, annot_tensor, in_pipe_meta))

        return image_tensor, annot_tensor, in_pipe_meta
