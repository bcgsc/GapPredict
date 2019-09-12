#!/bin/bash
if [ $# -lt 5 ]; then
	echo "Usage: $(basename $0) <gaps file> <sealer merged file> <output path> <sample size> <reads file>"
	exit 1
fi

gaps_file=$1;shift
merged_file=$1;shift
output_path=$1;shift
sample_size=$1;shift
reads_file=$1;shift

python3.6 ./1_search_by_sealer.py ${gaps_file} ${merged_file} ${output_path}
python3.6 ./2_search_by_not_sealer.py ${gaps_file} ${merged_file} ${output_path}
python3.6 ./3_random_sample.py ${output_path} ${sample_size}
./4_extract_reads_by_id.sh ${output_path}/human-scaffolds_fixed_gaps_flanks_random_sample_ids.txt ${output_path}/human-scaffolds_unfixed_gaps_flanks_random_sample_ids.txt ${output_path} ${reads_file}
python3.6 ./5_iterative_search.py ${output_path} ${merged_file}
python3.6 ./6_clean_flanks_with_N.py ${output_path}