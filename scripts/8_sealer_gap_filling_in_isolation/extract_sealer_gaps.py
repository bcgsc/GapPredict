import os
import sys

def next_n_lines(file, n):
    buf = [None]*n
    for i in range(n):
        line = file.readline()
        if line:
            buf[i] = line
        else:
            #can't return a complete n lines
            return None
    return buf

def strip_terminal_newline(string):
    if string[len(string) - 1] == "\n":
        return string[0:len(string)-1]
    else:
        return string

def extract_gap(sealed_file, gap_id, out_path):
    file = open(sealed_file, 'r')
    out_file = open(out_path + gap_id + "_gap.fasta", 'w+')
    sequence = next_n_lines(file, 2)
    bases = strip_terminal_newline(sequence[1])
    bases = bases[:len(bases) - 500][500:]

    out_file.write(sequence[0])
    out_file.write(bases)
    out_file.close()
    file.close()

sealer_out_dir = sys.argv[1]

sealer_out_path = sealer_out_dir + "/fixed/"
gaps = os.listdir(sealer_out_path)

for gap in gaps:
    filled_gap_path=sealer_out_path + gap + "/sealed/"
    merged_file = filled_gap_path + gap + ".sealer_merged.fa"
    merged_stat = os.stat(merged_file)
    size = merged_stat.st_size
    if size > 0:
        sealed_file = filled_gap_path + gap + ".sealer_scaffold.fa"
        extract_gap(sealed_file, gap, filled_gap_path)


sealer_out_path = sealer_out_dir + "/unfixed/"
gaps = os.listdir(sealer_out_path)

for gap in gaps:
    filled_gap_path=sealer_out_path + gap + "/sealed/"
    merged_file = filled_gap_path + gap + ".sealer_merged.fa"
    merged_stat = os.stat(merged_file)
    size = merged_stat.st_size
    if size > 0:
        sealed_file = filled_gap_path + gap + ".sealer_scaffold.fa"
        extract_gap(sealed_file, gap, filled_gap_path)
