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

def write_all_to_dest(dest_file, buf):
    for line in buf:
        dest_file.write(line)

def main():
    gaps_file = open(sys.argv[1], 'r')
    merged_file = open(sys.argv[2], 'r')
    output_file = open(sys.argv[3] + '/human-scaffolds_unfixed_gaps_flanks.fa', 'w+')

    gaps_map = {}

    pair = next_n_lines(gaps_file, 4)
    while pair is not None:
        id = parse_id(pair[0])
        left_flank_range = parse_range(pair[0])
        left_flank_right_endpoint = left_flank_range[1]
        map_id = id + "-" + str(left_flank_right_endpoint)
        if map_id not in gaps_map:
            gaps_map[map_id] = pair
        else:
            print("Error")
            return
        pair = next_n_lines(gaps_file, 4)

    sequence = next_n_lines(merged_file, 2)
    while sequence is not None:
        id = parse_merged_id(sequence[0])
        left_flank_right_endpoint = parse_merged_idx(sequence[0])
        map_id = id + "-" + left_flank_right_endpoint
        if map_id not in gaps_map:
            print("Error")
            return
        else:
            del gaps_map[map_id]
        sequence = next_n_lines(merged_file, 2)

    for key in gaps_map:
        pair = gaps_map[key]
        write_all_to_dest(output_file, pair)

    gaps_file.close()
    merged_file.close()
    output_file.close()



if __name__ == "__main__":
    main()