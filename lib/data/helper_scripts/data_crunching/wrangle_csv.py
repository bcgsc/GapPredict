import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as utils

correctness_threshold = 0.7
distance_threshold = 20
prediction_length = 750
primary_text_font_size = 55
secondary_text_font_size = 45

axiswidth = 3
ticksize = 15
mpl.rcParams['axes.linewidth'] = axiswidth
mpl.rcParams['xtick.major.size'] = ticksize
mpl.rcParams['xtick.major.width'] = axiswidth
mpl.rcParams['ytick.major.size'] = ticksize
mpl.rcParams['ytick.major.width'] = axiswidth

base_path = "E:\\Users\\Documents\\School_Year_18-19\\Term_1\\CPSC_449\\Sealer_NN\\lib\\data\\helper_scripts\\data_crunching\\"

def evaluate_gap(df, predict_from_left):
    query_coverage = []
    is_pass = []
    for i in range(len(df)):
        row = df.loc[i]

        prediction_bases = row['query_alignment_length_gap']

        gap_start = row["query_start_gap"]
        gap_end = row["query_end_gap"]
        if np.isnan(gap_start) or np.isnan(gap_end):
            query_coverage.append(0)
            is_pass.append(0)
            continue

        flank_start = row["query_start_flank"]
        flank_end = row["query_end_flank"]
        if np.isnan(flank_start) or np.isnan(flank_end):
            flank_start = -np.inf if predict_from_left else np.inf
            flank_end = -np.inf if predict_from_left else np.inf

        corrected_gap_end = max(gap_start, gap_end) if predict_from_left else min(gap_start, gap_end)
        corrected_flank_start = min(flank_start, flank_end) if predict_from_left else max(flank_start, flank_end)
        strict_gap_end = max(corrected_gap_end, corrected_flank_start) if predict_from_left else min(corrected_gap_end, corrected_flank_start)
        prediction_length_considered = strict_gap_end if predict_from_left else prediction_length - strict_gap_end
        coverage = prediction_bases/prediction_length_considered

        preliminary_pass = 1 if row.flank_correctness >= correctness_threshold else 0
        query_coverage.append(coverage)
        is_pass.append(preliminary_pass)
    coverage = pd.Series(data=np.array(query_coverage))
    gap_pass = pd.Series(data=np.array(is_pass))
    return coverage, gap_pass

def transform_dataframe(df, predict_from_left):
    df['total_percent_id_gap'] = (df.matches_gap/df.total_length_gap).fillna(0)
    df['target_gap_coverage'] = (df.target_alignment_length_gap/df.total_length_gap).fillna(0)
    df['target_gap_correctness'] = (df.total_percent_id_gap * df.target_gap_coverage).fillna(0)

    df['total_percent_id_flank'] = (df.matches_flank/df.total_length_flank).fillna(0)
    df['flank_coverage'] = (df.target_alignment_length_flank/df.target_length_flank).fillna(0)
    df['flank_correctness'] = df.total_percent_id_flank * df.flank_coverage
    gap_coverage, gap_pass = evaluate_gap(df, predict_from_left)

    df['query_gap_coverage'] = gap_coverage.fillna(0)
    df['gap_pass'] = gap_pass

def plot_comparison_scatter(data, file_path):
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 13)

    plt.figure(figsize=figure_dimensions)

    x = data["target_gap_correctness"] * 100
    y = data["target_correctness"] * 100

    sns.kdeplot(x, y)
    sns.kdeplot(x, y, cmap="Reds", shade=True, shade_lowest=False, cbar=True)
    plt.scatter(x, y, s=200, alpha=0.8)

    plt.xlim((-1, 101))
    plt.ylim((-1, 101))
    plt.xlabel('GapPredict correctness (%)')
    plt.ylabel('Sealer correctness (%)')

    plt.tight_layout()
    fig = plt.savefig(file_path)
    plt.close(fig)

def plot_scatter(data, file_path):
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 13)

    plt.figure(figsize=figure_dimensions)

    x = data["target_gap_correctness"]*100
    y = data["query_gap_coverage"]*100

    sns.kdeplot(x, y)
    ax = sns.kdeplot(x, y, cmap="Reds", shade=True, shade_lowest=False, cbar=True)
    ax.figure.axes[-1].tick_params(axis='both', which='major', labelsize=35)

    plt.scatter(x, y, alpha=0.8, s=200)

    plt.xlim((-1, 101))
    plt.ylim((-1, 101))
    plt.xlabel('Target correctness (%)')
    plt.ylabel('Query coverage (%)')

    plt.tight_layout()
    fig = plt.savefig(file_path)
    plt.close(fig)

def transform_sealer_dataframe(df):
    df['total_percent_id'] = (df.matches/df.total_length).fillna(0)
    df['coverage'] = (df.target_alignment_length/df.target_length).fillna(0)
    df['target_correctness'] = (df.total_percent_id * df.coverage).fillna(0)

def clean_sealer_dataframe(df):
    df['target_correctness'] = df.target_correctness.fillna(0)

def compute_metrics(gap_left, gap_right, left_subflank, right_subflank, id):
    gap_left_df = pd.read_csv(base_path + gap_left, index_col=0)
    gap_left_df.columns = ['gap_id', 'is_fixed', 'percent_identity_gap', 'query_start_gap', 'query_end_gap', 'query_length_gap', 'query_alignment_length_gap', 'target_start_gap', 'target_end_gap', 'target_length_gap', 'target_alignment_length_gap', 'total_bases_compared_gap', 'mismatches_gap', 'matches_gap', 'total_length_gap']
    gap_right_df = pd.read_csv(base_path + gap_right, index_col=0)
    gap_right_df.columns = ['gap_id', 'is_fixed', 'percent_identity_gap', 'query_start_gap', 'query_end_gap', 'query_length_gap', 'query_alignment_length_gap', 'target_start_gap', 'target_end_gap', 'target_length_gap', 'target_alignment_length_gap', 'total_bases_compared_gap', 'mismatches_gap', 'matches_gap', 'total_length_gap']
    left_subflank_right_df = pd.read_csv(base_path + left_subflank, index_col=0)
    left_subflank_right_df.columns = ['gap_id', 'is_fixed', 'percent_identity_flank', 'query_start_flank', 'query_end_flank', 'query_length_flank', 'query_alignment_length_flank', 'target_start_flank', 'target_end_flank', 'target_length_flank', 'target_alignment_length_flank', 'total_bases_compared_flank', 'mismatches_flank', 'matches_flank', 'total_length_flank']
    right_subflank_left_df = pd.read_csv(base_path + right_subflank, index_col=0)
    right_subflank_left_df.columns = ['gap_id', 'is_fixed', 'percent_identity_flank', 'query_start_flank', 'query_end_flank', 'query_length_flank', 'query_alignment_length_flank', 'target_start_flank', 'target_end_flank', 'target_length_flank', 'target_alignment_length_flank', 'total_bases_compared_flank', 'mismatches_flank', 'matches_flank', 'total_length_flank']

    lg_sum_p = pd.read_csv(base_path + "prediction_metrics.csv", index_col=0)
    gap_left_lsp = lg_sum_p[(lg_sum_p.is_left == 1) & (lg_sum_p.algorithm == id)]
    gap_right_lsp = lg_sum_p[(lg_sum_p.is_left == 0) & (lg_sum_p.algorithm == id)]

    gap_left_full_df = pd.merge(gap_left_df, right_subflank_left_df, on=['gap_id', 'is_fixed'], how='inner')
    gap_left_full_df = pd.merge(gap_left_full_df, gap_left_lsp, on=['gap_id', 'is_fixed'], how='inner')
    gap_right_full_df = pd.merge(gap_right_df, left_subflank_right_df, on=['gap_id', 'is_fixed'], how='inner')
    gap_right_full_df = pd.merge(gap_right_full_df, gap_right_lsp, on=['gap_id', 'is_fixed'], how='inner')

    assert len(gap_left_full_df) == len(gap_right_full_df)
    assert len(gap_left_full_df) == len(gap_left_df)

    transform_dataframe(gap_left_full_df, True)
    transform_dataframe(gap_right_full_df, False)

    columns_to_select = ['gap_id', 'is_fixed', 'flank_correctness', 'target_gap_correctness', 'query_gap_coverage', 'gap_pass', 'log-sum-probability']
    full_dataframe = gap_left_full_df.append(gap_right_full_df, ignore_index=True)
    columns_to_plot = full_dataframe[columns_to_select]

    fixed_pass = columns_to_plot[(columns_to_plot.is_fixed == 1) & (columns_to_plot.gap_pass == 1)]
    fixed_fail = columns_to_plot[(columns_to_plot.is_fixed == 1) & (columns_to_plot.gap_pass == 0)]
    total_fixed = len(fixed_pass) + len(fixed_fail)
    fixed_zero_pass = fixed_pass[(fixed_pass.query_gap_coverage < 0.05) & (fixed_pass.target_gap_correctness < 0.05)]
    fixed_zero_fail = fixed_fail[(fixed_fail.query_gap_coverage < 0.05) & (fixed_fail.target_gap_correctness < 0.05)]
    print("Fixed Pass = " + str(len(fixed_pass)) + "/" + str(total_fixed))
    print("Fixed Fail = " + str(len(fixed_fail)) + "/" + str(total_fixed))
    print("Fixed Zero Pass = " + str(len(fixed_zero_pass)) + "/" + str(len(fixed_pass)))
    print("Fixed Zero Fail = " + str(len(fixed_zero_fail)) + "/" + str(len(fixed_fail)))

    unfixed_pass = columns_to_plot[(columns_to_plot.is_fixed == 0) & (columns_to_plot.gap_pass == 1)]
    unfixed_fail = columns_to_plot[(columns_to_plot.is_fixed == 0) & (columns_to_plot.gap_pass == 0)]
    unfixed_zero_pass = unfixed_pass[(unfixed_pass.query_gap_coverage < 0.05) & (unfixed_pass.target_gap_correctness < 0.05)]
    unfixed_zero_fail = unfixed_fail[(unfixed_fail.query_gap_coverage < 0.05) & (unfixed_fail.target_gap_correctness < 0.05)]
    total_unfixed = len(unfixed_pass) + len(unfixed_fail)
    print("Unfixed Pass = " + str(len(unfixed_pass)) + "/" + str(total_unfixed))
    print("Unfixed Fail = " + str(len(unfixed_fail)) + "/" + str(total_unfixed))
    print("Unfixed Zero Pass = " + str(len(unfixed_zero_pass)) + "/" + str(len(unfixed_pass)))
    print("Unfixed Zero Fail = " + str(len(unfixed_zero_fail)) + "/" + str(len(unfixed_fail)))

    delimiter = utils.get_terminal_directory_character()

    out_path = base_path + "viz" + delimiter + id + delimiter
    os.makedirs(out_path, exist_ok=True)

    plot_scatter(fixed_pass, out_path + "fixed_pass.png")
    plot_scatter(fixed_fail, out_path + "fixed_fail.png")
    plot_scatter(unfixed_pass, out_path + "unfixed_pass.png")
    plot_scatter(unfixed_fail, out_path + "unfixed_fail.png")

    sealer_df = pd.read_csv(base_path + "sealer_merged.csv", index_col=0)
    transform_sealer_dataframe(sealer_df)
    fixed = full_dataframe[(full_dataframe.is_fixed == 1)]
    unfixed = full_dataframe[(full_dataframe.is_fixed == 0)]
    fixed_sealer = sealer_df[(sealer_df.is_fixed == 1)]
    unfixed_sealer = sealer_df[(sealer_df.is_fixed == 0)]

    fixed_sealer_merged = pd.merge(fixed, fixed_sealer, on=["gap_id", "is_fixed"], how="left")[['gap_id', 'is_fixed', 'target_gap_correctness', 'target_correctness']]
    unfixed_sealer_merged = pd.merge(unfixed, unfixed_sealer, on=["gap_id", "is_fixed"], how="left")[['gap_id', 'is_fixed', 'target_gap_correctness', 'target_correctness']]
    clean_sealer_dataframe(fixed_sealer_merged)
    clean_sealer_dataframe(unfixed_sealer_merged)

    plot_comparison_scatter(fixed_sealer_merged, out_path + "fixed_sealer_compare.png")
    plot_comparison_scatter(unfixed_sealer_merged, out_path + "unfixed_sealer_compare.png")

print("Beam Search")
compute_metrics("gap_left_prediction_data.csv", "gap_right_prediction_data.csv", "left_subflank_right_prediction_data.csv", "right_subflank_left_prediction_data.csv", "beam_search")
print()
print("Greedy")
compute_metrics("gap_left_prediction_data_greedy.csv", "gap_right_prediction_data_greedy.csv", "left_subflank_right_prediction_data_greedy.csv", "right_subflank_left_prediction_data_greedy.csv", "greedy")