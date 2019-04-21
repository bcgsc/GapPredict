import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if os.name == 'nt':
    base_path = "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\"
else:
    base_path = "/home/echen/Desktop/Projects/Sealer_NN/src/app/helper_scripts/data_crunching/"

def transform_dataframe(df):
    df['total_percent_id'] = (df.matches/df.total_length).fillna(0)
    df['coverage'] = (df.target_alignment_length/df.target_length).fillna(0)

def clean_dataframe(df):
    df['total_percent_id'] = df.total_percent_id.fillna(0)
    df['coverage'] = df.coverage.fillna(0)

def plot_scatter(data, file_path):
    primary_text_font_size = 45
    secondary_text_font_size = 35
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 13)

    plt.figure(figsize=figure_dimensions)

    ax = sns.scatterplot(x="total_percent_id", y="coverage", data=data, s=200)
    plt.xlim((-0.05, 1.05))
    plt.ylim((-0.05, 1.05))
    plt.xlabel('prediction identity')
    plt.ylabel('prediction coverage')

    plt.tight_layout()
    fig = plt.savefig(file_path)
    plt.close(fig)

sealer_df = pd.read_csv(base_path + "sealer_merged.csv", index_col=0)
gap_fill_df = pd.read_csv(base_path + "gap_left_prediction_data.csv", index_col=0)

transform_dataframe(sealer_df)

columns_to_select = ['gap_id', 'is_fixed', 'total_percent_id', 'coverage']
columns_to_plot = sealer_df[columns_to_select]

fixed = columns_to_plot[(columns_to_plot.is_fixed == 1)]
unfixed = columns_to_plot[(columns_to_plot.is_fixed == 0)]
joined_fixed = pd.merge(fixed, gap_fill_df, on=["gap_id", "is_fixed"], how="right") #TODO: maybe do an outer join to get the ones that didn't make it into sealer
joined_unfixed = pd.merge(unfixed, gap_fill_df, on=["gap_id", "is_fixed"], how="right")
clean_dataframe(joined_fixed)
clean_dataframe(joined_unfixed)

out_path = "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\sealer\\"
os.makedirs(out_path, exist_ok=True)

plot_scatter(joined_fixed, out_path + "sealer_fixed.png")
print("Fixed = " + str(len(joined_fixed)))
plot_scatter(joined_unfixed, out_path + "sealer_unfixed.png")
print("Unfixed = " + str(len(joined_unfixed)))