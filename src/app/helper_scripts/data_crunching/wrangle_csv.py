import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

    ax = sns.scatterplot(x="gap_correctness", y="gap_predict_tc", data=data, s=100)

    plt.tight_layout()
    fig = plt.savefig(file_path)
    plt.close(fig)

df = pd.read_csv("E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\gap_valid.csv", index_col=0)

headers = ["gap_left_pass", "gap_right_pass", "gap_left_correctness", "gap_right_correctness"]
additional_columns = []

for i in range(len(df)):
    row = df.loc[i]
    right_flank_correctness = row["right_subflank_pid"] * row["right_subflank_qc"]
    gap_left_pass = 1 if right_flank_correctness >= 0.7 else 0

    left_flank_correctness = row["left_subflank_pid"] * row["left_subflank_qc"]
    gap_right_pass = 1 if left_flank_correctness >= 0.7 else 0

    gap_left_correctness = row["gap_left_predict_pid"] * row["gap_left_predict_qc"]
    gap_right_correctness = row["gap_right_predict_pid"] * row["gap_right_predict_qc"]

    additional_columns.append([gap_left_pass, gap_right_pass, gap_left_correctness, gap_right_correctness])

additional_df = pd.DataFrame(data=np.array(additional_columns), columns=headers)
additional_df = additional_df.apply(pd.to_numeric)
merged_df = pd.concat([df, additional_df], axis=1)

independent_columns = ['gap_id', 'is_fixed', 'gap_pass', 'gap_predict_tc', 'gap_correctness']
left_seed = merged_df[['gap_id', 'is_fixed', 'gap_left_pass', 'gap_left_predict_tc', 'gap_left_correctness']]
right_seed = merged_df[['gap_id', 'is_fixed', 'gap_right_pass', 'gap_right_predict_tc', 'gap_right_correctness']]
left_seed.columns = independent_columns
right_seed.columns = independent_columns

independent_df = left_seed.append(right_seed, ignore_index=True)
fixed_pass = independent_df[(independent_df.is_fixed == 1) & (independent_df.gap_pass == 1)]
fixed_fail = independent_df[(independent_df.is_fixed == 1) & (independent_df.gap_pass == 0)]
total_fixed = len(fixed_pass) + len(fixed_fail)
print("Fixed Pass = " + str(len(fixed_pass)) + "/" + str(total_fixed))
print("Fixed Fail = " + str(len(fixed_fail)) + "/" + str(total_fixed))

unfixed_pass = independent_df[(independent_df.is_fixed == 0) & (independent_df.gap_pass == 1)]
unfixed_fail = independent_df[(independent_df.is_fixed == 0) & (independent_df.gap_pass == 0)]
total_unfixed = len(unfixed_pass) + len(unfixed_fail)
print("Unfixed Pass = " + str(len(unfixed_pass)) + "/" + str(total_unfixed))
print("Unfixed Fail = " + str(len(unfixed_fail)) + "/" + str(total_unfixed))

os.makedirs("E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\", exist_ok=True)

plot_scatter(fixed_pass, "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\fixed_pass.png")
plot_scatter(fixed_fail, "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\fixed_fail.png")
plot_scatter(unfixed_pass, "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\unfixed_pass.png")
plot_scatter(unfixed_fail, "E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\helper_scripts\\data_crunching\\viz\\unfixed_fail.png")