import argparse

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

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('-lf', nargs=1, help="left flank file", required=True)
    arg_parser.add_argument('-rf', nargs=1, help="right flank file", required=True)
    arg_parser.add_argument('-o', nargs=1, help="output file directory", required=True)
    arg_parser.add_argument('-l', nargs=1, help="length", required=True)

    args = arg_parser.parse_args()
    left_flank_path = args.lf[0]
    right_flank_path = args.rf[0]
    out_file_path = args.o[0]
    gap_path = args.g[0]
    length = int(args.l[0])

    if out_file_path[-1] != '/':
        out_file_path += '/'

    left_flank = open(left_flank_path, 'r')
    lf_out_file = open(out_file_path + "left_subflank.fasta", 'w+')
    right_flank = open(right_flank_path, 'r')
    rf_out_file = open(out_file_path + "right_subflank.fasta", 'w+')

    sequence = next_n_lines(left_flank, 2)
    lf_out_file.write(sequence[0])
    clean_flank = strip_terminal_newline(sequence[1])
    left_subflank = clean_flank[len(clean_flank) - length:]
    lf_out_file.write(left_subflank)

    sequence = next_n_lines(right_flank, 2)
    rf_out_file.write(sequence[0])
    clean_flank = strip_terminal_newline(sequence[1])
    right_subflank = clean_flank[:length]
    rf_out_file.write(right_subflank)

    left_flank.close()
    right_flank.close()
    lf_out_file.close()
    rf_out_file.close()


if __name__ == "__main__":
    main()