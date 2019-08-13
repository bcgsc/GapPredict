import os
import shutil
import sys

def strip_terminal_newline(string):
    if string[len(string) - 1] == "\n":
        return string[0:len(string)-1]
    else:
        return string

def main():
    dirty_file = open(sys.argv[1] + "/bad_gaps.txt")
    root_path = sys.argv[1] + '/dirty/'
    in_path = sys.argv[1] + '/'

    inner_paths = ["fixed/", "unfixed/"]

    dirty_set = set()

    for line in dirty_file:
        clean_line = strip_terminal_newline(line)
        dirty_set.add(clean_line)

    for inner_path in inner_paths:
        out_path = root_path + inner_path
        gap_path = in_path + inner_path
        all_gaps = os.listdir(gap_path)
        os.makedirs(out_path, exist_ok=True)

        for gap in all_gaps:
            if gap in dirty_set:
                shutil.move(gap_path + gap, out_path + gap)
