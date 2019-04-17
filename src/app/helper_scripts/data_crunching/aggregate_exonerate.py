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
    alignments = []
    for line in file:
        if "Command line" in line:
            continue
        elif "percent_identity" in line:
            p_identity = float(parse_metric(line))/100
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
        elif "total_bases_compared" in line:
            tbc = int(parse_metric(line))
        elif "mismatches" in line:
            mism = int(parse_metric(line))
        elif "matches" in line:
            matches = int(parse_metric(line))
            alignments.append({
                "pid": p_identity,
                "qs": q_start,
                "qe": q_end,
                "ql": q_length,
                "qal": q_alength,
                "ts": t_start,
                "te": t_end,
                "tl": t_length,
                "tal": t_alength,
                "tbc": tbc,
                "mism" : mism,
                "matches": matches
            })
    if len(alignments) == 0:
        return None
    if len(alignments) > 1:
        print(file_path)
    alignments.sort(key=lambda x:x["qs"])
    alignment_to_choose = alignments[0] if left_favour else alignments[-1]
    return alignment_to_choose

def to_row(data, gap_id, is_fixed):
    if data is None:
        pid = 0
        qs = None
        qe = None
        ql = 0
        qal = 0
        ts = None
        te = None
        tl = 0
        tal = 0
        tbc = 0
        mism = 0
        matches = 0
    else:
        pid = data["pid"]
        qs = data["qs"]
        qe = data["qe"]
        ql = data["ql"]
        qal = data["qal"]
        ts = data["ts"]
        te = data["te"]
        tl = data["tl"]
        tal = data["tal"]
        tbc = data["tbc"]
        mism = data["mism"]
        matches = data["matches"]
    return [gap_id, is_fixed, pid, qs, qe, ql, qal, ts, te, tl, tal, tbc, mism, matches]


def save_csv(data, path):
    csv_headers = ["gap_id", "is_fixed", "percent_identity", "query_start", "query_end", "query_length",
                   "query_alignment_length", "target_start", "target_end", "target_length", "target_alignment_length",
                   "total_bases_compared", "mismatches", "matches"]
    numeric_headers = csv_headers[1:]
    np_data = np.array(data)

    df = pd.DataFrame(data=np_data, columns=csv_headers)
    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
    df.to_csv(path)

gap_types = ["fixed", "unfixed"]
base_path = "/projects/btl/scratch/echen/real_align/validation/"
right_subflank_left_prediction_data = []
gap_left_prediction_data = []

left_subflank_right_prediction_data = []
gap_right_prediction_data = []
for gap_type in gap_types:
    is_fixed = 1 if gap_type == "fixed" else 0
    gaps_path = base_path + gap_type + "/"
    gaps = os.listdir(gaps_path)
    for gap in gaps:
        exonerate_outputs = gaps_path + gap + "/exonerate/"
        right_subflank_left_prediction = parse_exonerate_output(exonerate_outputs + "right_subflank_left_prediction.exn", left_favour=True)
        rslp_row = to_row(right_subflank_left_prediction, gap, is_fixed)
        right_subflank_left_prediction_data.append(rslp_row)

        gap_left_prediction = parse_exonerate_output(exonerate_outputs + "gap_left_prediction.exn", left_favour=True)
        gl_row = to_row(gap_left_prediction, gap, is_fixed)
        gap_left_prediction_data.append(gl_row)

        left_subflank_right_prediction = parse_exonerate_output(exonerate_outputs + "left_subflank_right_prediction.exn", left_favour=False)
        lsrp_row = to_row(left_subflank_right_prediction, gap, is_fixed)
        left_subflank_right_prediction_data.append(lsrp_row)

        gap_right_prediction = parse_exonerate_output(exonerate_outputs + "gap_right_prediction.exn", left_favour=False)
        gr_row = to_row(gap_right_prediction, gap, is_fixed)
        gap_right_prediction_data.append(gr_row)

save_csv(right_subflank_left_prediction_data, "/projects/btl/scratch/echen/real_align/right_subflank_left_prediction_data.csv")
save_csv(gap_left_prediction_data, "/projects/btl/scratch/echen/real_align/gap_left_prediction_data.csv")
save_csv(left_subflank_right_prediction_data, "/projects/btl/scratch/echen/real_align/left_subflank_right_prediction_data.csv")
save_csv(gap_right_prediction_data, "/projects/btl/scratch/echen/real_align/gap_right_prediction_data.csv")