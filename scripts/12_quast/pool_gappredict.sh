#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <output_dir output path> <gap_id_path file> <gappredict_flanks_and_gaps set 1 path>"
	exit 1
fi

output_dir=$1;shift
txt_path=$1;shift
gappadder_path=$1;shift
mkdir -p $output_dir

> $output_dir/set1_gappredict_forward.fasta
> $output_dir/set1_gappredict_rc.fasta
> $output_dir/set1_gappredict.fasta

while read gap_id; do
    cat $predictions_path/fixed/${gap_id}/predictions/top_one_forward_predict.fasta >> $output_dir/set1_gappredict_forward.fasta
    echo >> $output_dir/set1_gappredict_forward.fasta
done < $txt_path/left_intersection.txt

while read gap_id; do
    cat $predictions_path/fixed/${gap_id}/predictions/top_one_rc_predict.fasta >> $output_dir/set1_gappredict_rc.fasta
    echo >> $output_dir/set1_gappredict_rc.fasta
done < $txt_path/right_intersection.txt

cat $output_dir/set1_gappredict_forward.fasta $output_dir/set1_gappredict_rc.fasta > $output_dir/set1_gappredict.fasta
