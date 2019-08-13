#!/bin/bash

if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <read directory> <outpath>"
	exit 1
fi

mkdir -p ./log
read_dir=$1;shift
outpath=$1;shift

for i in {75..205..10}
  do
    nohup /usr/bin/time -pv ./abyss-bloom.slurm $i $outpath \
    ${read_dir}/NA12878_S1_L001_R1_001.fastq.gz \
    ${read_dir}/NA12878_S1_L001_R2_001.fastq.gz \
    ${read_dir}/NA12878_S1_L002_R1_001.fastq.gz \
    ${read_dir}/NA12878_S1_L002_R2_001.fastq.gz \
    &> ./log/k$i.log &
  done
