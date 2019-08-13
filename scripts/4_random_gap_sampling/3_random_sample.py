import random
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

def parse_id(id):
    id_copy = str(id)
    id_copy = strip_terminal_newline(id_copy)
    id = id_copy.split(":")[0].split(">")[1]
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

def main():
	ids = ["fixed", "unfixed"]
	for file_id in ids:
		gaps_file = open(sys.argv[1] + '/human-scaffolds_' + file_id + '_gaps_flanks.fa', 'r')
		output_file = open(sys.argv[1] + '/human-scaffolds_' + file_id + '_gaps_flanks_random_sample.fa', 'w+')
		id_file = open(sys.argv[1] +  '/human-scaffolds_' + file_id + '_gaps_flanks_random_sample_ids.txt', 'w+')

		max_flank_length = 500
		random_sample_length = int(sys.argv[2])

		pair = next_n_lines(gaps_file, 4)
		pairs = []
		while pair is not None:
			id = parse_id(pair[0])
			left_flank_range = parse_range(pair[0])
			right_flank_range = parse_range(pair[2])
			left_flank_right_endpoint = left_flank_range[1]
			right_flank_left_endpoint = right_flank_range[0]

			left_flank_length = abs(int(left_flank_range[1]) - int(left_flank_range[0]))
			right_flank_length = abs(int(right_flank_range[1]) - int(right_flank_range[0]))
			estimated_gap_length = abs(int(right_flank_left_endpoint) - int(left_flank_right_endpoint))

			if left_flank_length != max_flank_length or right_flank_length != max_flank_length or estimated_gap_length < 20:
				pair = next_n_lines(gaps_file, 4)
				continue

			pairs.append(pair)
			pair = next_n_lines(gaps_file, 4)

		random.seed(0)
		random.shuffle(pairs)

		random_sample = pairs[:random_sample_length]

		for i in range(len(random_sample)):
			pair = random_sample[i]
			is_last = True if i == len(random_sample) - 1 else False
			write_all_to_dest(output_file, pair)
			write_ids_to_dest(id_file, pair, is_last)

		gaps_file.close()
		output_file.close()
		id_file.close()



if __name__ == "__main__":
    main()