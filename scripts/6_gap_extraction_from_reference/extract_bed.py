import argparse
import pandas as pd
import numpy as np

def log_error(error_file, gap_id):
    file = open(error_file, 'a')
    file.write(gap_id + '\n')
    file.close()

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('-o', nargs=1, help="output directory", required=True)
    arg_parser.add_argument('-bed', nargs=1, help="Flanks BED file", required=True)
    arg_parser.add_argument('-id', nargs=1, help="gap ID", required=True)
    arg_parser.add_argument('-e', nargs=1, help='error log path', required=True)
    args = arg_parser.parse_args()
    output_directory = args.o[0]
    bed_file = args.bed[0]
    gap_id = args.id[0]
    error_file = args.e[0]
    df = pd.read_csv(bed_file, index_col=False, header=None, sep="\t")
    print(gap_id)
    try:
        assert len(df) == 2
    except AssertionError:
        print("Strange mapping")
        log_error(error_file, gap_id)
        return
    left_flank = df.loc[0]
    right_flank = df.loc[1]
    try:
        assert len(left_flank) == 6
        assert len(right_flank) == 6
        assert left_flank[2] < right_flank[1]
    except AssertionError:
        print("Flank format error")
        log_error(error_file, gap_id)
        return
    gap_row = np.array([[0, left_flank[0], left_flank[2], right_flank[1], gap_id, left_flank[4], left_flank[5]]])
    gap_df = pd.DataFrame(data=gap_row[:, 1:], index=gap_row[:,0])
    gap_df.to_csv(output_directory, index=False, header=None, sep='\t')


if __name__ == "__main__":
    main()
