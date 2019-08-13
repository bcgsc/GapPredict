import os
import pandas as pd
import numpy as np
import sys

def strip_terminal_newline(string):
    if string[len(string) - 1] == "\n":
        return string[0:len(string)-1]
    else:
        return string

def parse_metric(string):
    return strip_terminal_newline(string).split(":")[1]

def parse_cigar(string):
    tokens = string.split(' ')[1:]
    length = 0
    for i in range(len(tokens)):
        if i % 2 == 0:
            assert tokens[i] == "M" or tokens[i] == "D" or tokens[i] == "I"
        else:
            length += int(tokens[i])
    return length

def parse_exonerate_output(file_path):
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
        elif "cigar" in line:
            cigar = parse_metric(line)
            total_length = parse_cigar(cigar)
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
                "mism": mism,
                "matches": matches,
                "total_length": total_length
            })
    if len(alignments) == 0:
        return None
    if len(alignments) != 1:
        print(file_path)
    alignment_to_choose = alignments[0]
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
        total_length = 0
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
        total_length = data["total_length"]
    return [gap_id, is_fixed, pid, qs, qe, ql, qal, ts, te, tl, tal, tbc, mism, matches, total_length]


def save_csv(data, path):
    csv_headers = ["gap_id", "is_fixed", "percent_identity", "query_start", "query_end", "query_length",
                   "query_alignment_length", "target_start", "target_end", "target_length", "target_alignment_length",
                   "total_bases_compared", "mismatches", "matches", "total_length"]
    numeric_headers = csv_headers[1:]
    np_data = np.array(data)

    df = pd.DataFrame(data=np_data, columns=csv_headers)
    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
    df.to_csv(path)

base_path = sys.argv[1]
out_path = sys.argv[2]

gap_types = ["fixed", "unfixed"]
sealer_data = []
sealer_merged_data = []
for gap_type in gap_types:
    is_fixed = 1 if gap_type == "fixed" else 0
    gaps_path = base_path + gap_type + "/"
    gaps = os.listdir(gaps_path)
    for gap in gaps:
        sealer_fill_file = gaps_path + gap + "/sealer.exn"
        sealer_fill = parse_exonerate_output(sealer_fill_file)
        sealer_fill_row = to_row(sealer_fill, gap, is_fixed)
        sealer_data.append(sealer_fill_row)

        sealer_merged_file = gaps_path + gap + "/sealer_merged.exn"
        sealer_merge = parse_exonerate_output(sealer_merged_file)
        sealer_merge_row = to_row(sealer_merge, gap, is_fixed)
        sealer_merged_data.append(sealer_merge_row)

save_csv(sealer_data, out_path + "/sealer.csv")
save_csv(sealer_merged_data, out_path + "/sealer_merged.csv")