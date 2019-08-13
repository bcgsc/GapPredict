#!/bin/bash
if [ $# -lt 4 ]; then
	echo "Usage: $(basename $0) <k> <fixed reads dir> <unfixed reads dir> <current working dir>"
	exit 1
fi
k=$1;shift
fixed_data_dir=$1;shift
unfixed_data_dir=$1;shift
curr_dir=$1;shift

cd $fixed_data_dir
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  base_out_path=${curr_dir}/sealer_out/fixed/$gap_id
  raw_reads=$fixed_data_dir/$gap_id/$gap_id.fastq
  mkdir -p $base_out_path
  ${curr_dir}/abyss-bloom.slurm $k $base_out_path $raw_reads > $base_out_path/k$k.log
done

cd $unfixed_data_dir
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  base_out_path=${curr_dir}/sealer_out/unfixed/$gap_id
  raw_reads=$unfixed_data_dir/$gap_id/$gap_id.fastq
  mkdir -p $base_out_path
  ${curr_dir}/abyss-bloom.slurm $k $base_out_path $raw_reads > $base_out_path/k$k.log
done