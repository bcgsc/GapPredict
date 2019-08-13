#!/bin/bash
#SBATCH --job-name=sealer_lstm
#SBATCH --partition=all
#SBATCH --ntasks=48
#SBATCH --mem=350G
#SBATCH --exclusive
#SBATCH --output=log/o_%j.out
#SBATCH -N 1-1

if [ $# -ne 2 ]; then
	echo "Usage: $(basename $0) <Draft scaffolds FASTA> <Bloom filter directory>"
	exit 1
fi

set -x
set -e

draft=$1; shift
bloom_filters=$1; shift
draft_base=human-scaffolds

/usr/bin/time -pv abyss-sealer -v -S $draft \
	-t $draft_base-sealed-trace.txt \
	-o $draft_base'.sealer' \
	-L 150 \
	-j 48 \
	-P 10  \
	-k 75 --input-bloom=<(gunzip -c ${bloom_filters}/k75'.bloom.gz') \
	-k 85 --input-bloom=<(gunzip -c ${bloom_filters}/k85'.bloom.gz') \
	-k 95 --input-bloom=<(gunzip -c ${bloom_filters}/k95'.bloom.gz') \
	-k 105 --input-bloom=<(gunzip -c ${bloom_filters}/k105'.bloom.gz') \
	-k 115 --input-bloom=<(gunzip -c ${bloom_filters}/k115'.bloom.gz') \
	-k 125 --input-bloom=<(gunzip -c ${bloom_filters}/k125'.bloom.gz') \
	-k 135 --input-bloom=<(gunzip -c ${bloom_filters}/k135'.bloom.gz') \
	-k 145 --input-bloom=<(gunzip -c ${bloom_filters}/k145'.bloom.gz') \
	-k 155 --input-bloom=<(gunzip -c ${bloom_filters}/k155'.bloom.gz') \
	-k 165 --input-bloom=<(gunzip -c ${bloom_filters}/k165'.bloom.gz') \
	-k 175 --input-bloom=<(gunzip -c ${bloom_filters}/k175'.bloom.gz') \
	-k 185 --input-bloom=<(gunzip -c ${bloom_filters}/k185'.bloom.gz') \
	-k 195 --input-bloom=<(gunzip -c ${bloom_filters}/k195'.bloom.gz') \
	-k 205 --input-bloom=<(gunzip -c ${bloom_filters}/k205'.bloom.gz') 
	
	

### EOF ###
