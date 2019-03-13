import os

import utils.directory_utils as UTILS
import math

def strip_terminal_newline(string):
    if string[-1] == "\n":
        return string[:len(string)-1]
    else:
        return string

def format_file(path):
    txt_file_name = "align.txt"
    formatted_file_name = "formatted_align.txt"
    file = open(path + txt_file_name, 'r')
    formatted_file = open(path + formatted_file_name, 'w+')

    for i in range(2):
        line = file.readline()
        formatted_file.write(line)

    actual_seq = strip_terminal_newline(file.readline())
    match_string = strip_terminal_newline(file.readline())
    predict_seq = strip_terminal_newline(file.readline())

    length = len(actual_seq)

    assert (len(actual_seq) == len(match_string) and len(match_string) == len(predict_seq))
    line_length = 100

    iterations = math.ceil(length/line_length)
    for i in range(iterations):
        formatted_file.write(actual_seq[i*100:(i+1)*100] + "\n")
        formatted_file.write(match_string[i*100:(i+1)*100] + "\n")
        formatted_file.write(predict_seq[i*100:(i+1)*100] + "\n")
        formatted_file.write("\n")

    for i in range(6):
        line = file.readline()
        formatted_file.write(line)

    file.close()
    formatted_file.close()

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    rnn_dim_directories = os.listdir(root)
    replicates = 2

    ids = set()
    for rnn_dim in rnn_dim_directories:
        dim_path = root + rnn_dim + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    lf = "left_flank"
    rf = "right_flank"
    f = "forward"
    rc = "reverse_complement"

    flanks = [lf, rf]
    strands = [f, rc]

    for id in ids:
        for i in range(replicates):
            for rnn_dim in rnn_dim_directories:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char

                for flank in flanks:
                    for strand in strands:
                        path = folder_path + flank + terminal_char + strand + terminal_char
                        format_file(path)

if __name__ == "__main__":
    main()
