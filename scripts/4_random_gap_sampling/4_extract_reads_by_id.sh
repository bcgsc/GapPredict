#!/bin/bash
if [ $# -lt 4 ]; then
	echo "Usage: $(basename $0) <fixed random sample ids> <unfixed random sample ids> <output path> <all reads>"
	exit 1
fi

fixed_id_path=$1; shift
unfixed_id_path=$1; shift
output_path=$1; shift
fastq_path=$1; shift

gunzip -c $fastq_path | grep -A 3 --no-group-separator -F -f $fixed_id_path > ${output_path}/human-scaffolds_fixed_gaps_flanks_random_sample.fastq
gunzip -c $fastq_path | grep -A 3 --no-group-separator -F -f $unfixed_id_path > ${output_path}/human-scaffolds_unfixed_gaps_flanks_random_sample.fastq
