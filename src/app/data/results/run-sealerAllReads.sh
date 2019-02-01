#!/bin/bash
#SBATCH --job-name=Sanctuary_sealer
#SBATCH --partition=all
#SBATCH --ntasks=48
#SBATCH --mem=350G
#SBATCH --exclusive
#SBATCH --output=log/o_%j.out
#SBATCH -N 1-1

export PATH=/projects/btl/benv/arch/centos6/abyss-2.1.1/maxk160/bin:$PATH

if [ $# -ne 1 ]; then
	echo "Usage: $(basename $0) <Draft scaffolds FASTA>"
	exit 1
fi

set -x
set -e

draft=$1; shift

draft_basename=$(basename $draft)
export draft_basename
draft_base=$(perl -e 'if($ENV{'draft_basename'} =~ /(\S+)\.fa/){print "$1";} ')


/usr/bin/time -pv abyss-sealer -v -S $draft \
	-t $draft_base-sealed-trace.txt \
	-o $draft_base'.sealer' \
	-L 150 \
	-j 48 \
	-P 10  \
	-k 75 --input-bloom=<(zcat BFs/k75'.bloom.z') \
	-k 85 --input-bloom=<(zcat BFs/k85'.bloom.z') \
	-k 95 --input-bloom=<(zcat BFs/k95'.bloom.z') \
	-k 105 --input-bloom=<(zcat BFs/k105'.bloom.z') \
	-k 115 --input-bloom=<(zcat BFs/k115'.bloom.z') \
	-k 125 --input-bloom=<(zcat BFs/k125'.bloom.z') 
	

### EOF ###
