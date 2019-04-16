import os

import numpy as np
import pandas as pd


def strip_terminal_newline(string):
    if string[len(string) - 1] == "\n":
        return string[0:len(string)-1]
    else:
        return string

def parse_metric(string):
    return strip_terminal_newline(string).split(":")[1]

def parse_exonerate_output(file_path, left_favour=True):
    file = open(file_path, 'r')
    p_identity = -1
    q_start = -1
    q_end = -1
    q_length = -1
    q_alength = -1
    t_start = -1
    t_end = -1
    t_length = -1
    t_alength = -1
    iterations = 0
    for line in file:
        if "Command line" in line:
            continue
        elif "percent_identity" in line:
            p_identity = float(parse_metric(line))/100
            iterations += 1
        elif "query_start" in line:
            q_start = int(parse_metric(line))
        elif "query_end" in line:
            q_end = int(parse_metric(line))
        elif "query_length" in line:
            q_length = int(parse_metric(line))
        elif "query_alignment_length" in line:
            q_alength = int(parse_metric(line))
        elif "target_start" in line:
            t_start = int(parse_metric(line))
        elif "target_end" in line:
            t_end = int(parse_metric(line))
        elif "target_length" in line:
            t_length = int(parse_metric(line))
        elif "target_alignment_length" in line:
            t_alength = int(parse_metric(line))
            if left_favour:
                break
    if iterations == 0:
        return None
    if iterations > 1:
        print(file_path)
    return {
        "pid": p_identity,
        "qs": q_start,
        "qe": q_end,
        "ql": q_length,
        "qal": q_alength,
        "ts": t_start,
        "te": t_end,
        "tl": t_length,
        "tal": t_alength
    }


gap_types = ["fixed", "unfixed"]
base_path = "/projects/btl/scratch/echen/real_align/validation/"
data = []
for gap_type in gap_types:
    is_fixed = 1 if gap_type == "fixed" else 0
    gaps_path = base_path + gap_type + "/"
    gaps = os.listdir(gaps_path)
    for gap in gaps:
        exonerate_outputs = gaps_path + gap + "/exonerate/"
        right_subflank_left_prediction = parse_exonerate_output(exonerate_outputs + "right_subflank_left_prediction.exn", left_favour=True)
        if right_subflank_left_prediction is None:
            right_subflank_pid = 0
            right_subflank_query_coverage = 0
            right_subflank_start = -1
        else:
            right_subflank_pid = right_subflank_left_prediction["pid"]
            right_subflank_query_coverage = right_subflank_left_prediction["qal"] / right_subflank_left_prediction["ql"]
            right_subflank_start = min(right_subflank_left_prediction["ts"], right_subflank_left_prediction["te"])

        gap_left_prediction = parse_exonerate_output(exonerate_outputs + "gap_left_prediction.exn", left_favour=True)
        if gap_left_prediction is None:
            gap_left_pid = 0
            gap_left_query_coverage = 0
            gap_left_target_coverage = 0
        else:
            gap_left_pid = gap_left_prediction["pid"]
            gap_left_query_coverage = gap_left_prediction["qal"] / gap_left_prediction["ql"]
            gap_left_end = max(gap_left_prediction["ts"], gap_left_prediction["te"])
            left_predict_target_coverage_end = max(gap_left_end, right_subflank_start)
            left_predict_target_coverage_length = left_predict_target_coverage_end
            gap_left_target_coverage = gap_left_prediction["qal"] / left_predict_target_coverage_length

        left_subflank_right_prediction = parse_exonerate_output(exonerate_outputs + "left_subflank_right_prediction.exn", left_favour=False)
        if left_subflank_right_prediction is None:
            left_subflank_pid = 0
            left_subflank_query_coverage = 0
            left_subflank_start = 751
        else:
            left_subflank_pid = left_subflank_right_prediction["pid"]
            left_subflank_query_coverage = left_subflank_right_prediction["qal"] / left_subflank_right_prediction["ql"]
            left_subflank_start = max(left_subflank_right_prediction["ts"], left_subflank_right_prediction["te"])

        gap_right_prediction = parse_exonerate_output(exonerate_outputs + "gap_right_prediction.exn", left_favour=False)
        if gap_right_prediction is None:
            gap_right_pid = 0
            gap_right_query_coverage = 0
            gap_right_target_coverage = 0
        else:
            gap_right_pid = gap_right_prediction["pid"]
            gap_right_query_coverage = gap_right_prediction["qal"] / gap_right_prediction["ql"]
            gap_right_end = min(gap_right_prediction["ts"], gap_right_prediction["te"])
            right_predict_target_coverage_end = min(gap_right_end, left_subflank_start)
            right_predict_target_coverage_length = 750 - right_predict_target_coverage_end
            gap_right_target_coverage = gap_right_prediction["qal"]/right_predict_target_coverage_length

        row = [gap, is_fixed, right_subflank_pid, right_subflank_query_coverage, gap_left_pid, gap_left_query_coverage, gap_left_target_coverage, left_subflank_pid, left_subflank_query_coverage, gap_right_pid, gap_right_query_coverage, gap_right_target_coverage]
        data.append(row)
csv_headers = ["gap_id", "is_fixed", "right_subflank_pid", "right_subflank_qc", "gap_left_predict_pid", "gap_left_predict_qc", "gap_left_predict_tc", "left_subflank_pid", "left_subflank_qc", "gap_right_predict_pid", "gap_right_predict_qc", "gap_right_predict_tc"]
numeric_headers = csv_headers[1:]
np_data = np.array(data)

df = pd.DataFrame(data=np_data, columns=csv_headers)
df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
df.to_csv("/projects/btl/scratch/echen/real_align/gap_valid.csv")
