#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <output_dir output path> <gap_id_path file> <reference_flanks_and_gaps set 1 path>"
	exit 1
fi

output_dir=$1;shift
txt_path=$1;shift
gappadder_path=$1;shift
mkdir -p $output_dir

> $output_dir/set1_references.fasta

while read gap_id; do
    cat $reference_path/${gap_id}/gap/${gap_id}_hg38_aln_gap.fasta >> $output_dir/set1_references.fasta
done < $txt_path/intersection.txt
