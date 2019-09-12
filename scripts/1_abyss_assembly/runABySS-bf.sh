#!/bin/bash

export TMPDIR=/var/tmp

if [ $# -lt 4 ]; then
	echo "Usage: $(basename $0) <kmer size> <num threads> <output prefix> <NA12878 directory> <output dir>" >&2
	exit 1
fi

set -eux -o pipefail

k=$1; shift
kc=3
bf=100G
j=$1; shift
prefix=$1; shift
read_dir=$1; shift
output_dir=$1; shift

dir=${output_dir}/k${k}/kc${kc}
mkdir -p $dir
cd $dir

export lib='pelib1 pelib2'

export pelib1="${read_dir}/NA12878_S1_L001_R1_001.fastq.gz ${read_dir}/NA12878_S1_L001_R2_001.fastq.gz"
export pelib2="${read_dir}/NA12878_S1_L002_R1_001.fastq.gz ${read_dir}/NA12878_S1_L002_R2_001.fastq.gz"

# run the assembly

which abyss-pe

/usr/bin/time -pv abyss-pe    \
	j=$j k=$k kc=$kc B=$bf l=40 s=1000 v=-v q=15 H=4 S=1000-10000 N=9 \
	name=${prefix} \
	$@
