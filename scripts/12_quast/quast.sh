#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <quast path> <output_dir output path> <pooled_gaps path>"
	exit 1
fi

quast_path=$1;shift
output_dir=$1;shift
pooled_path=$1;shift

mkdir -p $output_dir/gappredict_pass/set1
mkdir -p $output_dir/gappredict_pass/set1f
mkdir -p $output_dir/gappredict_pass/set1rc
mkdir -p $output_dir/sealer/set1
mkdir -p $output_dir/gappadder/set1

python $quast_path -o $output_dir/sealer/set1 -r $pooled_path/reference/set1_references.fasta -m 0 -t 8 /projects/btl/scratch/echen/quast_intersection/sealer/set1_sealer.fasta

python $quast_path -o $output_dir/gappadder/set1 -r $pooled_path/reference/set1_references.fasta -m 0 -t 8 $pooled_path/gappadder/set1_gappadder.fasta

python $quast_path -o $output_dir/gappredict_pass/set1 -r $pooled_path/reference/set1_references.fasta -m 0 -t 8 $pooled_path/gappredict_pass/set1_gappredict.fasta

python $quast_path -o $output_dir/gappredict_pass/set1f -r $pooled_path/reference/set1_references.fasta -m 0 -t 8 $pooled_path/gappredict_pass/set1_gappredict_forward.fasta

python $quast_path -o $output_dir/gappredict_pass/set1rc -r $pooled_path/reference/set1_references.fasta -m 0 -t 8 $pooled_path/gappredict_pass/set1_gappredict_rc.fasta
