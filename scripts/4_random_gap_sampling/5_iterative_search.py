import random
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

def parse_range(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    range = id_copy.split(":")[1]
    idx = range.split("-")
    return int(idx[0]), int(idx[1])

def parse_contig_id(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    id = id_copy.split(":")[0].split(">")[1]
    return id

def parse_full_id(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    id = id_copy.split(">")[1]
    return id

def write_all_to_dest(dest_file, buf):
    for line in buf:
        dest_file.write(line)

def write_ids_to_dest(dest_file, buf, is_last):
    for i in range(len(buf)):
        if i % 2 == 0:
            full_id = str(buf[i]).split(">")[1]
            if is_last and len(buf) - i <= 2:
                full_id = strip_terminal_newline(full_id)
            dest_file.write(full_id)

def parse_merged_id(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    id = id_copy.split("_")[0].split(">")[1]
    return id

def parse_merged_idx(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    id = id_copy.split("_")[1]
    return id

def main():
	ids = ["fixed", "unfixed"]
	for file_id in ids:
		fasta_file = open(sys.argv[1] + '/human-scaffolds_' + file_id + '_gaps_flanks_random_sample.fa', 'r')
		reads_file = open(sys.argv[1] + '/human-scaffolds_' + file_id + '_gaps_flanks_random_sample.fastq', 'r')
		output_directory = sys.argv[1] + '/out/' + file_id + '/'

		reads = []
		read = next_n_lines(reads_file, 4)
		while read is not None:
			reads.append(read)
			read = next_n_lines(reads_file, 4)

		reads_file.close()

		gaps_map = {}
		if fixed:
			merged_file = open(sys.argv[2], 'r')
			sequence = next_n_lines(merged_file, 2)
			while sequence is not None:
				id = parse_merged_id(sequence[0])
				left_flank_right_endpoint = parse_merged_idx(sequence[0])
				map_id = id + "-" + left_flank_right_endpoint
				if map_id not in gaps_map:
					gaps_map[map_id] = sequence
				else:
					print("Error")
					return
				sequence = next_n_lines(merged_file, 2)
			merged_file.close()

		pair = next_n_lines(fasta_file, 4)
		while pair is not None:
			contig_id = parse_contig_id(pair[0])
			left_flank_range = parse_range(pair[0])
			right_flank_range = parse_range(pair[2])
			left_flank_left_endpoint = left_flank_range[0]
			right_flank_right_endpoint = right_flank_range[1]

			full_id = contig_id + "_" + str(left_flank_left_endpoint) + "-" + str(right_flank_right_endpoint)
			new_folder = output_directory + full_id + "/"
			os.makedirs(new_folder, exist_ok=True)

			left_flank_id = parse_full_id(pair[0])
			right_flank_id = parse_full_id(pair[2])

			fasta_output = open(new_folder + full_id + ".fasta", "w+")
			fastq_output = open(new_folder + full_id + ".fastq", "w+")

			write_all_to_dest(fasta_output, pair)

			if fixed:
				left_flank_right_endpoint = left_flank_range[1]
				map_id = contig_id + "-" + str(left_flank_right_endpoint)
				if map_id not in gaps_map:
					print("Error")
					return
				else:
					sequence = gaps_map[map_id]
					write_all_to_dest(fasta_output, sequence)

			for read in reads:
				read_id = read[0]
				if left_flank_id in read_id or right_flank_id in read_id:
					write_all_to_dest(fastq_output, read)

			fasta_output.close()
			fastq_output.close()

			pair = next_n_lines(fasta_file, 4)

		fasta_file.close()



if __name__ == "__main__":
    main()