#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <HG38 path> <draft assembly path>"
	exit 1
fi

reference=$1;shift
draft_assembly=$1;shift
for gap_class in */; do
  for directory in $gap_class*/; do
    gap_id=$(echo $directory | cut -d'/' -f2)
    base_path=$gap_class$gap_id
    raw_fasta=$base_path/$gap_id.fasta
    
    flanks_path=$base_path/flanks
    mkdir -p $flanks_path
    flanks_file=$flanks_path/$gap_id'_'flanks.fasta 
    left_flank_file=$flanks_path/$gap_id'_'left_flank.fasta
    right_flank_file=$flanks_path/$gap_id'_'right_flank.fasta
    head -4 $raw_fasta > $flanks_file
    head -2 $raw_fasta > $left_flank_file
    head -4 $raw_fasta | tail -2 > $right_flank_file
    
    align_id=$gap_id'_'hg38_aln
    
    bwa_path=$base_path/bwa
    mkdir -p $bwa_path
    bam_file=$bwa_path/$align_id.bam
    sorted_bam_file=$bwa_path/$align_id.sorted.bam
    bwa mem -t 20 -x intractg $reference $flanks_file 2>$bwa_path/bwa_mem.log.txt | samtools view -bS - > $bam_file
    samtools sort $bam_file 1>$sorted_bam_file
    samtools index $sorted_bam_file
    
    bed_path=$base_path/bed
    mkdir -p $bed_path
    bed_file=$bed_path/$align_id'_'flanks.bed
    merged_bed_file=$bed_path/$align_id'_'merged_flanks.bed
    bedtools bamtobed -i $sorted_bam_file > $bed_file 
    bedtools merge -i $bed_file -d 100000 > $merged_bed_file
    
    gaps_bed_file=$bed_path/$align_id'_'gaps.bed
    gap_file=$bed_path
    python3.6 ./extract_bed.py -bed $bed_file -o $gaps_bed_file -id $gap_id -e ./bad_gaps.txt
    
    gap_path=$base_path/gap
    mkdir -p $gap_path
    full_gap_file=$gap_path/$align_id'_'full_gap.fasta
    gap_file=$gap_path/$align_id'_'gap.fasta
    bedtools getfasta -fi $reference -bed $merged_bed_file > $full_gap_file
    bedtools getfasta -fi $reference -bed $gaps_bed_file > $gap_file
    
    #draft genome alignment
    bwa_draft=$base_path/bwa_draft
    mkdir -p $bwa_draft
    bam_file=$bwa_draft/$align_id.bam
    sorted_bam_file=$bwa_draft/$align_id.sorted.bam
    bwa mem -t 20 -x intractg $draft_assembly $flanks_file 2>$bwa_draft/bwa_mem.log.txt | samtools view -bS - > $bam_file
    samtools sort $bam_file 1>$sorted_bam_file
    samtools index $sorted_bam_file
    
    bed_draft_path=$base_path/bed_draft
    mkdir -p $bed_draft_path
    bed_file=$bed_draft_path/$align_id'_'flanks.bed
    merged_bed_file=$bed_draft_path/$align_id'_'merged_flanks.bed
    bedtools bamtobed -i $sorted_bam_file > $bed_file 
    bedtools merge -i $bed_file -d 1000 > $merged_bed_file
    
    gap_draft_path=$base_path/gap_draft
    mkdir -p $gap_draft_path
    full_gap_file=$gap_draft_path/$align_id'_'full_gap.fasta
    bedtools getfasta -fi $draft_assembly -bed $merged_bed_file > $full_gap_file
  done
done
directory=$(pwd)
python3.6 ./filter_dirty_files.py ${directory}
