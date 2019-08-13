#!/bin/bash 
#SBATCH --job-name=sealer_lstm
#SBATCH --partition=all
#SBATCH --ntasks=16 
#SBATCH --mem=120G
#SBATCH --output=log/o_%j.out 
#SBATCH -N 1-1

# Based on Ben's job script for making bloom filters for Konnector.
# Modified by Ka Ming Nip on August 29, 2014.

if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <k> <output path> <paths to reads>"
	exit 1
fi

set -ex

echo "Job started at $(date)"

echo "PATH for job was:"
echo $PATH

k=$1; shift
outpath=$1; shift
abyss-bloom build -v -v -k$k -j16 -b100G -l2 -q15 - $@ |gzip -c> ${outpath}/k${k}.bloom.gz

echo "Job ended at $(date)"
