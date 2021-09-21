#!/bin/bash
PATH=/gsc/btl/linuxbrew/bin/:$PATH

output_dir=/projects/btl/scratch/echen/quast/gappredict_pass
mkdir -p $output_dir

> $output_dir/set1_gappredict_forward.fasta
> $output_dir/set1_gappredict_rc.fasta
> $output_dir/set1_gappredict.fasta
> $output_dir/set2_gappredict_forward.fasta
> $output_dir/set2_gappredict_rc.fasta
> $output_dir/set2_gappredict.fasta

txt_path=/projects/btl/scratch/echen/quast
predictions_path=/projects/btl/scratch/echen/real_align/validation

while read gap_id; do
    cat $predictions_path/fixed/${gap_id}/predictions/top_one_forward_predict.fasta >> $output_dir/set1_gappredict_forward.fasta
    echo >> $output_dir/set1_gappredict_forward.fasta
done < $txt_path/left_fixed.txt

while read gap_id; do
    cat $predictions_path/fixed/${gap_id}/predictions/top_one_rc_predict.fasta >> $output_dir/set1_gappredict_rc.fasta
    echo >> $output_dir/set1_gappredict_rc.fasta
done < $txt_path/right_fixed.txt

while read gap_id; do
    cat $predictions_path/unfixed/${gap_id}/predictions/top_one_forward_predict.fasta >> $output_dir/set2_gappredict_forward.fasta
    echo >> $output_dir/set2_gappredict_forward.fasta
done < $txt_path/left_unfixed.txt

while read gap_id; do
    cat $predictions_path/unfixed/${gap_id}/predictions/top_one_rc_predict.fasta >> $output_dir/set2_gappredict_rc.fasta
    echo >> $output_dir/set2_gappredict_rc.fasta
done < $txt_path/right_unfixed.txt

cat $output_dir/set1_gappredict_forward.fasta $output_dir/set1_gappredict_rc.fasta > $output_dir/set1_gappredict.fasta
cat $output_dir/set2_gappredict_forward.fasta $output_dir/set2_gappredict_rc.fasta > $output_dir/set2_gappredict.fasta