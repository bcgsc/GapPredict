#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <alignment output path> <scaffold file> <reads path>"
	exit 1
fi

alignment_path=$1;shift
scaffold_file=$1;shift
reads_path=$1;shift

mkdir -p ${alignment_path}/bwa

bwa index ${scaffold_file}

bwa mem -t 32 ${scaffold_file} ${reads_path}/NA12878_S1_L001_R1_001.fastq.gz ${reads_path}/NA12878_S1_L001_R2_001.fastq.gz 2>${alignment_path}/bwa_mem_L001.log.txt | samtools view -bS - > ${alignment_path}/bwa/reads_L001.bam

bwa mem -t 32 ${scaffold_file} ${reads_path}/NA12878_S1_L002_R1_001.fastq.gz ${reads_path}/NA12878_S1_L002_R2_001.fastq.gz 2>${alignment_path}/bwa_mem_L002.log.txt | samtools view -bS - > ${alignment_path}/bwa/reads_L002.bam

samtools sort ${alignment_path}/bwa/reads_L001.bam 1> ${alignment_path}/bwa/reads_L001.sorted.bam
samtools sort ${alignment_path}/bwa/reads_L002.bam 1> ${alignment_path}/bwa/reads_L002.sorted.bam
samtools index ${alignment_path}/bwa/reads_L001.sorted.bam
samtools index ${alignment_path}/bwa/reads_L002.sorted.bam
samtools flagstats ${alignment_path}/bwa/reads_L001.sorted.bam 1> ${alignment_path}/bwa/reads_L001_flagstats.txt
samtools flagstats ${alignment_path}/bwa/reads_L002.sorted.bam 1> ${alignment_path}/bwa/reads_L002_flagstats.txt
samtools stats ${alignment_path}/bwa/reads_L001.sorted.bam 1> ${alignment_path}/bwa/reads_L001_stats.txt
samtools stats ${alignment_path}/bwa/reads_L002.sorted.bam 1> ${alignment_path}/bwa/reads_L002_stats.txt
