#!/bin/bash 
#SBATCH --job-name=sealer_lstm
#SBATCH --partition=all
#SBATCH --ntasks=16 
#SBATCH --mem=120G
#SBATCH --output=log/o_%j.out 
#SBATCH -N 1-1

# Based on Ben's job script for making bloom filters for Konnector.
# Modified by Ka Ming Nip on August 29, 2014.

if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <k> <paths to reads>"
	exit 1
fi

set -ex

echo "Job started at $(date)"

export PATH=/projects/btl/benv/arch/centos6/abyss-2.1.1/maxk160/bin:$PATH

echo "PATH for job was:"
echo $PATH

k=$1; shift
abyss-bloom build -v -v -k$k -j16 -b100G -l2 -q15 - $@ |gzip -c>/projects/btl/scratch/echen/bloom_filters/k${k}.bloom.gz

echo "Job ended at $(date)"
