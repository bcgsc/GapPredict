import argparse

COMPLEMENT_MAP = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G"
}

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

def reverse_complement(sequence):
    reverse_seq = sequence[::-1]
    reverse_complement_seq = ""
    for i in range(len(reverse_seq)):
        reverse_complement_seq += COMPLEMENT_MAP[reverse_seq[i].upper()]
    return reverse_complement_seq

def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('-lf', nargs=1, help="left flank prediction file path", required=True)
    arg_parser.add_argument('-rf', nargs=1, help="right flank prediction file path", required=True)
    args = arg_parser.parse_args()
    left_flank_file_path = args.lf[0]
    right_flank_file_path = args.rf[0]

    left_flank_in = open(left_flank_file_path, 'r')
    right_flank_in = open(right_flank_file_path, 'r')

    left_flank_sequence = next_n_lines(left_flank_in, 2)
    left_flank_in.close()

    right_flank_sequence = next_n_lines(right_flank_in, 2)
    right_flank_in.close()

    left_flank_out = open(left_flank_file_path, 'w+')
    right_flank_out = open(right_flank_file_path, 'w+')

    flank_length=500

    left_flank_out.write(left_flank_sequence[0])
    left_flank_out.write(strip_terminal_newline(left_flank_sequence[1][flank_length:]))

    right_flank_out.write(right_flank_sequence[0])
    right_flank_rc = reverse_complement(strip_terminal_newline(right_flank_sequence[1]))
    right_flank_out.write(right_flank_rc[:len(right_flank_rc)-flank_length])

    left_flank_out.close()
    right_flank_out.close()


if __name__ == "__main__":
    main()
