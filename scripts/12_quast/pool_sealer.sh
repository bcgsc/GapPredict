#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <output_dir output path> <gap_id_path file> <sealer_flanks_and_gaps set 1 path>"
	exit 1
fi

output_dir=$1;shift
txt_path=$1;shift
gappadder_path=$1;shift
mkdir -p $output_dir

> $output_dir/set1_sealer.fasta

while read gap_id; do
  sealer_merged=$sealer_path/$gap_id/sealed/$gap_id'.sealer_merged.fa'
  if [ -f $sealer_merged ]
  then
    cat $sealer_merged >> $output_dir/set1_sealer.fasta
  else
    echo "missing set 1 ${gap_id}"
  fi
done < $txt_path/intersection.txt
