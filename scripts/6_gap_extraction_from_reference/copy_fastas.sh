#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <fixed gap flank FASTAs> <unfixed gap flank FASTAs>"
	exit 1
fi
out_path=$(pwd)
mkdir -p $out_path/fixed
mkdir -p $out_path/unfixed

fixed_flanks=$1; shift
unfixed_flanks=$1; shift
cd ${fixed_flanks}
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  base_path=$gap_id
  raw_fasta=$base_path/$gap_id.fasta
  mkdir $out_path/fixed/$gap_id
  cp $raw_fasta $out_path/fixed/$gap_id/$gap_id.fasta
done

cd ${unfixed_flanks}
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  base_path=$gap_id
  raw_fasta=$base_path/$gap_id.fasta
  mkdir $out_path/unfixed/$gap_id
  cp $raw_fasta $out_path/unfixed/$gap_id/$gap_id.fasta
done

cd ${out_path}