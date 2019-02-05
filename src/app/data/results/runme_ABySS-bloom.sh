#!/bin/bash
for i in {75..155..10}
  do
    nohup /usr/bin/time -pv /projects/btl/scratch/echen/scripts/abyss-bloom.slurm $i \
    /projects/btl/datasets/hsapiens/NA12878-Illumina/NA12878_S1_L001_R1_001.fastq.gz \
    /projects/btl/datasets/hsapiens/NA12878-Illumina/NA12878_S1_L001_R2_001.fastq.gz \
    /projects/btl/datasets/hsapiens/NA12878-Illumina/NA12878_S1_L002_R1_001.fastq.gz \
    /projects/btl/datasets/hsapiens/NA12878-Illumina/NA12878_S1_L002_R2_001.fastq.gz \
    &> /projects/btl/scratch/echen/bloom_filters/log/k$i.log &
  done
