#!/bin/bash
if [ $# -lt 4 ]; then
	echo "Usage: $(basename $0) <GapPredict lib directory> <FASTA for a single gap> <FASTQ for a single gap> <output directory>"
	exit 1
fi

lib_dir=$1; shift
fasta=$1; shift
fastq=$1; shift
outdir=$1; shift
python3.6 ${lib_dir}/full_keras_pipeline.py -fa ${fasta} -fq ${fastq} -o ${outdir}