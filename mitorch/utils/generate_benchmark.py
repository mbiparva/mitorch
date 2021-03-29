import argparse
from copy import deepcopy
import functools
import glob
import os
from pathlib import Path
import time
import yaml
import numpy as np

################################################################################
### ROBUSTNESS TRANSFORMS
################################################################################

# for most transforms, convert to tensor first if torchio else convert after
rob_trfs = {
    'Anisotropy': {
        'severity_controllers': [
            {
                'downsampling_range': [float(i) for i in np.linspace(2., 10., 5)]
            }
        ]
    },
    'BiasField': {
        'severity_controllers': [
            {
                'coefficients_range': [float(i) for i in np.linspace(0., 1.5, 6)[1:]]
            }
        ]
    },
    'Blur': {
        'severity_controllers': [
            {
                'std': [float(i) for i in np.linspace(0., 4., 6)[1:]]
            }
        ]
    },
    'ContrastCompression': {
        'severity_controllers': [
            {
                'adjuster.gamma': np.linspace(1., 0.3, 6)[1:]
            }
        ]
    },
    'ContrastExpansion': {
        'severity_controllers': [
            {
                'adjuster.gamma': np.linspace(1., 3., 6)[1:].tolist()
            }
        ]
    },
    'Downsample': {
        'severity_controllers': [
            {
                'spacing_transform.pixdim': [float(i) for i in np.linspace(1., 4., 6)[1:]]
            }
        ]
    },
    'DownsampleKeepSize': {
        'severity_controllers': [
            {
                'spacing_transform.pixdim': [float(i) for i in np.linspace(1., 4., 6)[1:]]
            }
        ]
    },
    'ElasticDeformation': {
        'severity_controllers': [
            {
                'max_displacement': [float(x) for x in np.linspace(0., 30., 6)[1:]]
            }
        ]
    },
    'Ghosting': {
        'severity_controllers': [
            {
                'num_ghosts_range': [float(x) for x in [3, 5, 7, 9, 11]],
                'intensity_range': [float(x) for x in np.linspace(0.0, 2.5, 6)[1:]]
            }
        ]
    },
    'Motion': {
        'severity_controllers': [
            {
                'degrees_range': [float(i) for i in np.linspace(0.0, 5.0, 6)[1:]],
                'translation_range': [float(i) for i in np.linspace(0.0, 10.0, 6)[1:]],
                'num_transforms': [2, 4, 6, 8, 10]
            }
        ]
    },
    'RicianNoise': {
        'severity_controllers': [
            {
                'std': np.linspace(0., 0.1, 6)[1:].tolist()
            }
        ]
    },
    'Rotate': {
        'severity_controllers': [
            {
                f'range_{axis}': [float(x) for x in np.linspace(0.0, np.pi/2, 6)[1:]]
                for axis in 'xyz'
            }
        ]
    },
    'Upsample': {
        'severity_controllers': [
            {
                'spacing_transform.pixdim': [float(i) for i in np.linspace(1., 0.5, 6)[1:]]
            }
        ]
    }
}

################################################################################
### LOOP OVER DATASET
################################################################################

transforms_dict = dict()

for rob_transform, settings in rob_trfs.items():
    print("-" * 10, rob_transform)
    transforms_dict[rob_transform] = dict()
    sv_controller = settings['severity_controllers'][0]
    for param, values in sv_controller.items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if isinstance(values, list) and isinstance(values[0], tuple):
                values = [list(value) for value in values]
            if isinstance(values, list) and isinstance(values[0], np.ndarray):
                values = [value.tolist() for value in values]
            transforms_dict[rob_transform][param] = values
            print(param, values)

with open(r'generate_benchmark.yaml', 'w') as file:
    yaml.dump(transforms_dict, file, default_flow_style=False, encoding='UTF-8')
