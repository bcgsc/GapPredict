import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as utils

base_path = "E:\\Users\\Documents\\School_Year_18-19\\Term_1\\CPSC_449\\Sealer_NN\\lib\\data\\helper_scripts\\data_crunching\\"

axiswidth = 3
ticksize = 15
mpl.rcParams['axes.linewidth'] = axiswidth
mpl.rcParams['xtick.major.size'] = ticksize
mpl.rcParams['xtick.major.width'] = axiswidth
mpl.rcParams['ytick.major.size'] = ticksize
mpl.rcParams['ytick.major.width'] = axiswidth

def transform_dataframe(df):
    df['total_percent_id'] = (df.matches/df.total_length).fillna(0)
    df['coverage'] = (df.target_alignment_length/df.target_length).fillna(0)
    df['target_correctness'] = (df.total_percent_id * df.coverage * 100).fillna(0)

def plot_box(data, file_path):
    primary_text_font_size = 55
    secondary_text_font_size = 45
    linewidth = 6
    rotation = 0
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
    plt.ylim((-5, 101))
    plt.xlabel('Set Classification')
    plt.ylabel('Target correctness (%)')

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
joined_fixed["fixed_status"] = "Set 1"
joined_unfixed = pd.merge(unfixed, gap_fill_df, on=["gap_id", "is_fixed"], how="right")
joined_unfixed = joined_unfixed[~np.isnan(joined_unfixed.target_correctness)]
joined_unfixed["fixed_status"] = "Set 2"
full_data = joined_fixed.append(joined_unfixed, ignore_index=True)

delimiter = utils.get_terminal_directory_character()

out_path = base_path + "viz" + delimiter + "sealer" + delimiter
os.makedirs(out_path, exist_ok=True)

plot_box(full_data, out_path + "sealer.png")
print("Fixed = " + str(len(joined_fixed)))
print("Unfixed = " + str(len(joined_unfixed)))

print("Set 1 Mean: " + str(joined_fixed["target_correctness"].mean()))
print("Set 1 Median: " + str(joined_fixed["target_correctness"].median()))
print("Set 2 Mean: " + str(joined_unfixed["target_correctness"].mean()))
print("Set 2 Median: " + str(joined_unfixed["target_correctness"].median()))

gap_id = ['gap_id']
fixed_intersection = pd.merge(fixed, gap_fill_df, on=["gap_id", "is_fixed"])
id_col = fixed_intersection[gap_id]
id_col.to_csv("E:\\Users\\Documents\\sealer_set1.txt");