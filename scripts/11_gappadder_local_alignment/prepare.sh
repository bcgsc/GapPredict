#!/bin/bash
if [ $# -lt 1 ]; then
	echo "Usage: $(basename $0) <picked_seqs.fa>"
	exit 1
fi

picked_seqs=$1;shift

bwa index ${picked_seqs}
