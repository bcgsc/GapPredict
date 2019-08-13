import os
import shutil
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

def main():
	ids = ["fixed", "unfixed"]
	for file_id in ids:
		base_path = sys.argv[1] + '/out/'
		fasta_directory = base_path + file_id + '/'
		dirty_directory = base_path + 'dirty/' + file_id + '/'

		gaps = os.listdir(fasta_directory)
		dirty_flanks = []

		for gap in gaps:
			fasta_file = base_path + file_id + '/' + gap + '/' + gap + '.fasta'
			file = open(fasta_file)
			flanks = next_n_lines(file, 4)

			left_flank = strip_terminal_newline(flanks[1])
			right_flank = strip_terminal_newline(flanks[3])

			if 'N' in left_flank or 'N' in right_flank or 'n' in left_flank or 'n' in right_flank:
				dirty_flanks.append(gap)

			file.close()

		print("Dirty Flanks: " + str(len(dirty_flanks)))

		for gap in dirty_flanks:
			src_directory = base_path + file_id + '/' + gap + '/'
			shutil.move(src_directory, dirty_directory)



if __name__ == "__main__":
    main()