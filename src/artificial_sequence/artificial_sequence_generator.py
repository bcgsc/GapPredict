import math
import os

import numpy as np

from preprocess.SequenceReverser import SequenceReverser

if os.name == 'nt':
    root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\data\\artificial\\'
else:
    root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/data/artificial/'

if not os.path.exists(root_path):
    os.makedirs(root_path)

read_length = 250
gap_length = 350
flank_length = 500
coverage = 60

bases = np.array(["A", "C", "G", "T"])

reverser = SequenceReverser()


def get_total_length():
    return gap_length + flank_length * 2 + read_length * 2


def write_sequence_to_fasta(id, sequence, file):
    file.write(">" + id + "\n")
    file.write(sequence + "\n")


def write_fasta(file_name, full_seq, seq_of_interest, left_flank, right_flank, gap):
    file = open(root_path + file_name, "w+")
    write_sequence_to_fasta("full_seq", full_seq, file)
    write_sequence_to_fasta("seq_of_interest", seq_of_interest, file)
    write_sequence_to_fasta("left_flank", left_flank, file)
    write_sequence_to_fasta("right_flank", right_flank, file)
    write_sequence_to_fasta("gap", gap, file)
    file.close()

def dissect_sequence(sequence):
    length = get_total_length()
    seq_of_interest = sequence[read_length:length - read_length]
    left_flank = seq_of_interest[:flank_length]
    right_flank = seq_of_interest[gap_length + flank_length:]
    gap = seq_of_interest[flank_length:gap_length+flank_length]
    return seq_of_interest, left_flank, right_flank, gap

def generate_repetitive_sequence(motif):
    motif_length = len(motif)
    length = get_total_length()
    repeats = math.ceil(length / motif_length)
    full_sequence = ""
    for i in range(repeats):
        full_sequence += motif

    full_sequence = full_sequence[:length]

    seq_of_interest, left_flank, right_flank, gap = dissect_sequence(full_sequence)

    write_fasta("repetitive.fasta", full_sequence, seq_of_interest, left_flank, right_flank, gap)
    generate_fastq_reads("repetitive.fastq", full_sequence)

def generate_random_sequence():
    length = get_total_length()
    random_bases = np.random.randint(0, 4, length)

    full_sequence = "".join(bases[random_bases])

    seq_of_interest, left_flank, right_flank, gap = dissect_sequence(full_sequence)

    write_fasta("random.fasta", full_sequence, seq_of_interest, left_flank, right_flank, gap)
    generate_fastq_reads("random.fastq", full_sequence)


def extract_reads(sequence, idx):
    read = sequence[idx:idx + read_length]
    pair = sequence[idx + read_length:idx + read_length * 2]
    reverse_complement_pair = reverser.reverse_complement(pair)
    return read, reverse_complement_pair


def write_fastq(sequence, id, file):
    file.write(str(id) + "\n")
    file.write(sequence + "\n")
    file.write("+\n")
    file.write("\n")

def generate_fastq_reads(fastq_name, sequence):
    length = get_total_length()
    num_reads = math.ceil((length * coverage) / read_length)
    left_flank_right_endpoint = read_length + flank_length
    right_flank_left_endpoint = read_length + flank_length + gap_length

    possible_indexes = np.concatenate([np.arange(left_flank_right_endpoint),
                                      np.arange(right_flank_left_endpoint - read_length, length - read_length)])
    sample_idx = np.random.randint(0, len(possible_indexes), num_reads)

    file = open(root_path + fastq_name, "w+")

    read_id = 0

    for idx in sample_idx:
        read, reverse_complement_pair = extract_reads(sequence, idx)
        write_fastq(read, read_id, file)
        read_id += 1
        write_fastq(reverse_complement_pair, read_id, file)
        read_id += 1

    file.close()

generate_random_sequence()
generate_repetitive_sequence("ATGC")