#!/usr/bin/env python
# coding: utf-8

import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import seaborn as sn


current_path = os.path.abspath('')

exp_path = os.path.normpath(os.path.join(current_path, '..', 'wmh_pytorch', 'experiments', 'SRIBILTest'))

exp_id = [
    '20210331_011836_530936',
    '20210402_041942_989499',
][-1]


exp_path_full = os.path.join(exp_path, exp_id)

df_file_path = os.path.join(exp_path_full, 'exp_ls_results.pkl')

assert os.path.exists(df_file_path)

with open(df_file_path, 'rb') as fh:
    exp_df = pkl.load(fh)


def plot_one_exp(exp_curr):
    exp_id_curr, exp_desc_curr, exp_df_curr = exp_curr

    print('plotting', exp_id_curr)

    x_ind = exp_df_curr.index.to_numpy().astype(int)

    y_mean = exp_df_curr.dice_coefficient_mean.to_numpy()

    y_std = exp_df_curr.dice_coefficient_std.to_numpy()

    fig = plt.figure(figsize=(16, 9), dpi= 300, facecolor='w', edgecolor='k')
    ax = plt.gca()

    ax.fill_between(x_ind, y_mean-y_std, y_mean+y_std, facecolor='dodgerblue', edgecolor='black', linewidth=2, antialiased=False, interpolate=False, alpha=0.20)
    ax.plot(x_ind, y_mean, marker='v', markerfacecolor='orange', markersize=12, c='r', linewidth=3)

    ax.grid()
    ax.set_xlim([x_ind[0], x_ind[-1]])
    ax.set_ylim(0, 1)
    ax.set_title(f"transformation: {exp_desc_curr[0]['t_name']}\nParameters: {exp_desc_curr[0]['t_params']}")

    fig.savefig(os.path.join(exp_path_full, f"{exp_id}_{exp_desc_curr[0]['t_name']}"))

    fig.savefig(os.path.join(f"{exp_id}_{exp_desc_curr[0]['t_name']}"))


def main():
    for e in exp_df:
        plot_one_exp(e)


if '__name__' == '__main__':
    main()
