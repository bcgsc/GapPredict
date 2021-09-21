#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <output_dir output path> <gap_id_path file> <gappadder_flanks_and_gaps set 1 path>"
	exit 1
fi

output_dir=$1;shift
txt_path=$1;shift
gappadder_path=$1;shift
mkdir -p $output_dir

> $output_dir/set1_gappadder.fasta

while read gap_id; do
    full_gap=$gappadder_path/$gap_id/'gappadder_full_gap.fa'
    if [ -f $full_gap ]
    then
      cat $full_gap >> $output_dir/set1_gappadder.fasta
    else
      echo "missing set 1 ${gap_id}"
    fi
done < $txt_path/intersection.txt
