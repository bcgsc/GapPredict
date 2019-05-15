import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as utils

if os.name == 'nt':
    base_path = "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\lib\\app\\helper_scripts\\data_crunching\\"
else:
    base_path = "/home/echen/Desktop/Projects/Sealer_NN/lib/app/helper_scripts/data_crunching/"

def transform_dataframe(df):
    df['total_percent_id'] = (df.matches/df.total_length).fillna(0)
    df['coverage'] = (df.target_alignment_length/df.target_length).fillna(0)
    df['target_correctness'] = (df.total_percent_id * df.coverage * 100).fillna(0)

def plot_box(data, file_path):
    primary_text_font_size = 45
    secondary_text_font_size = 35
    linewidth = 6
    rotation = 45
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 13)

    plt.figure(figsize=figure_dimensions)

    flierprops = {
        'markersize': 10,
        'markerfacecolor': 'red',
        'marker': 'o'
    }

    ax = sns.boxplot(data=data, x="fixed_status", y="target_correctness", linewidth=linewidth, flierprops=flierprops)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    plt.ylim((-5, 100))
    plt.xlabel('filled_with_all_reads')
    plt.ylabel('target % correctness')

    plt.tight_layout()
    fig = plt.savefig(file_path)
    plt.close(fig)

sealer_df = pd.read_csv(base_path + "sealer_merged.csv", index_col=0)
gap_fill_df = pd.read_csv(base_path + "gap_left_prediction_data.csv", index_col=0)

transform_dataframe(sealer_df)

columns_to_select = ['gap_id', 'is_fixed', 'target_correctness']
columns_to_plot = sealer_df[columns_to_select]

fixed = columns_to_plot[(columns_to_plot.is_fixed == 1)]
unfixed = columns_to_plot[(columns_to_plot.is_fixed == 0)]
joined_fixed = pd.merge(fixed, gap_fill_df, on=["gap_id", "is_fixed"], how="right")
joined_fixed = joined_fixed[~np.isnan(joined_fixed.target_correctness)]
joined_fixed["fixed_status"] = "filled"
joined_unfixed = pd.merge(unfixed, gap_fill_df, on=["gap_id", "is_fixed"], how="right")
joined_unfixed = joined_unfixed[~np.isnan(joined_unfixed.target_correctness)]
joined_unfixed["fixed_status"] = "unfilled"
full_data = joined_fixed.append(joined_unfixed, ignore_index=True)

delimiter = utils.get_terminal_directory_character()

out_path = base_path + "viz" + delimiter + "sealer" + delimiter
os.makedirs(out_path, exist_ok=True)

plot_box(full_data, out_path + "sealer.png")
print("Fixed = " + str(len(joined_fixed)))
print("Unfixed = " + str(len(joined_unfixed)))