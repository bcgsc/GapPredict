#!/bin/bash

if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <flanks and gaps directory> <indexed picked_seqs.fa> <out dir>"
	exit 1
fi

flanks_and_gaps=$1;shift
picked_seqs=$1;shift
out_dir=$1;shift

fix=fixed
valid_dir=${out_dir}/align/$fix
mkdir -p $valid_dir

current_flanks=${flanks_and_gaps}/${fix}
cd $current_flanks

for gap in */; do
    gap_id=$(echo $gap | cut -d'/' -f1)
    mkdir -p ${valid_dir}/${gap_id}
    cp ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_full_gap.fasta ${valid_dir}/${gap_id}
    cp ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_gap.fasta ${valid_dir}/${gap_id}
    
    bwa mem -t 16 $picked_seqs ${valid_dir}/${gap_id}/${gap_id}_hg38_aln_full_gap.fasta 2> ${valid_dir}/${gap_id}/bwa_mem.log.txt | samtools view -bS - > ${valid_dir}/${gap_id}/gap.bam
    
    samtools view ${valid_dir}/${gap_id}/gap.bam | awk '{print $3}' | head -1 | grep -f - -A 1 $picked_seqs > ${valid_dir}/${gap_id}/gappadder_full_gap.fa
    
    samtools view ${valid_dir}/${gap_id}/gap.bam | awk '{print $10}' | head -1 > ${valid_dir}/${gap_id}/query.txt
    
    exonerate ${valid_dir}/${gap_id}/gappadder_full_gap.fa ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_gap.fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --bestn 1 > ${valid_dir}/${gap_id}/align.exn
done
    
fix=unfixed
valid_dir=${out_dir}/align/$fix
mkdir -p $valid_dir

current_flanks=${flanks_and_gaps}/${fix}
cd $current_flanks

for gap in */; do
    gap_id=$(echo $gap | cut -d'/' -f1)
    mkdir -p ${valid_dir}/${gap_id}
    cp ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_full_gap.fasta ${valid_dir}/${gap_id}
    cp ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_gap.fasta ${valid_dir}/${gap_id}
    
    bwa mem -t 16 $picked_seqs ${valid_dir}/${gap_id}/${gap_id}_hg38_aln_full_gap.fasta 2> ${valid_dir}/${gap_id}/bwa_mem.log.txt | samtools view -bS - > ${valid_dir}/${gap_id}/gap.bam
    
    samtools view ${valid_dir}/${gap_id}/gap.bam | awk '{print $3}' | head -1 | grep -f - -A 1 $picked_seqs > ${valid_dir}/${gap_id}/gappadder_full_gap.fa
    
    samtools view ${valid_dir}/${gap_id}/gap.bam | awk '{print $10}' | head -1 > ${valid_dir}/${gap_id}/query.txt
    
    exonerate ${valid_dir}/${gap_id}/gappadder_full_gap.fa ${current_flanks}/${gap_id}/gap/${gap_id}_hg38_aln_gap.fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --bestn 1 > ${valid_dir}/${gap_id}/align.exn
done

#python3.6 ${curr_dir}/aggregate_exonerate_gappadder.py ${curr_dir}/align/ ${curr_dir}/csv/
