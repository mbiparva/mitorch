#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2021 Mahdi Biparva, mahdi.biparva@gmail.com
#  miTorch: Medical Imaging with PyTorch
#  Deep Learning Package for 3D medical imaging in PyTorch
#  Implemented by Mahdi Biparva, April 2021
#  Brain Imaging Lab, Sunnybrook Research Institute (SRI)

import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import seaborn as sn
import sys


def plot_one_exp(exp_curr):
    _, exp_desc_curr, exp_df_curr = exp_curr

    x_ind = exp_df_curr.index.to_numpy().astype(int)

    y_mean = exp_df_curr.dice_coefficient_mean.to_numpy()

    y_std = exp_df_curr.dice_coefficient_std.to_numpy()

    fig = plt.figure(figsize=(16, 9), dpi= 300, facecolor='w', edgecolor='k')
    ax = plt.gca()

    ax.fill_between(x_ind, y_mean-y_std, y_mean+y_std, facecolor='dodgerblue', edgecolor='black', linewidth=2,
                    antialiased=False, interpolate=False, alpha=0.20)
    ax.plot(x_ind, y_mean, marker='v', markerfacecolor='orange', markersize=12, c='r', linewidth=3)

    ax.grid()
    ax.set_xlim([x_ind[0], x_ind[-1]])
    ax.set_ylim(0, 1)
    ax.set_title(f"transformation: {exp_desc_curr[0]['t_name']}\nParameters: {exp_desc_curr[0]['t_params']}")

    fig.savefig(os.path.join(exp_path_full, f"{exp_id}_{exp_desc_curr[0]['t_name']}"))

    # fig.savefig(os.path.join(f"{exp_id}_{exp_desc_curr[0]['t_name']}"))

    plt.close('all')


def main():
    print(f'loading {df_file_path}')
    for exp_curr in exp_df:
        exp_id_curr, exp_desc_curr, exp_df_curr = exp_curr

        print(f'plotting {exp_id_curr}|{len(exp_df)}: {exp_desc_curr}')

        plot_one_exp(exp_curr)


if __name__ == '__main__':
    exp_id = [
        '20210402_041942_989499',  # 1x1
        '20210402_042100_825498'  # 1x4
        '20210402_042340_020114'  # 4x4
        '20210402_042422_704225'  # 4x4 es
    ][0]

    if len(sys.argv) == 2:
        exp_id = sys.argv[1]

    current_path = os.path.abspath('')

    exp_path = os.path.normpath(os.path.join(current_path, '..', '..', 'experiments', 'SRIBILTest'))

    exp_path_full = os.path.join(exp_path, exp_id)

    df_file_path = os.path.join(exp_path_full, 'exp_ls_results.pkl')

    assert os.path.exists(df_file_path)

    with open(df_file_path, 'rb') as fh:
        exp_df = pkl.load(fh)

    main()
