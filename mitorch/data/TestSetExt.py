#  Copyright (c) 2020 Mahdi Biparva, mahdi.biparva@sri.utoronto.ca
#  miTorch Deep Learning Package
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, May 2020
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import torch
import numpy as np
from data.SRIBILSet import SRIBIL


class TestSet(SRIBIL):
    def __init__(self, cfg, mode, transform):
        super().__init__(cfg, mode, transform)
        self.prefix_name = True

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
