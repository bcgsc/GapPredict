#!/bin/bash
#SBATCH --job-name=sealer_lstm
#SBATCH --partition=all
#SBATCH --ntasks=48
#SBATCH --mem=350G
#SBATCH --exclusive
#SBATCH --output=log/o_%j.out
#SBATCH -N 1-1
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <fixed flanks and gaps dir> <unfixed flanks and gaps dir>"
	exit 1
fi

echo "Job started at $(date)"

curr_dir=$(pwd)
fixed_dir=$1;shift
unfixed_dir=$1;shift

cd $fixed_dir
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  bloom_path=${curr_dir}/sealer_out/fixed/$gap_id
  out_path=$bloom_path/sealed
  
  mkdir -p $out_path
  
  gap_fasta=$fixed_dir/$gap_id/gap_draft/$gap_id'_'hg38_aln_full_gap.fasta
  abyss-sealer -v -S $gap_fasta \
	  -t $out_path/$gap_id-sealed-trace.txt \
	  -o $out_path/$gap_id'.sealer' \
	  -L 150 \
	  -j 48 \
	  -P 10  \
	  -k 75 --input-bloom=<(gunzip -c $bloom_path/k75'.bloom.gz') \
	  -k 85 --input-bloom=<(gunzip -c $bloom_path/k85'.bloom.gz') \
	  -k 95 --input-bloom=<(gunzip -c $bloom_path/k95'.bloom.gz') \
	  -k 105 --input-bloom=<(gunzip -c $bloom_path/k105'.bloom.gz') \
	  -k 115 --input-bloom=<(gunzip -c $bloom_path/k115'.bloom.gz') \
	  -k 125 --input-bloom=<(gunzip -c $bloom_path/k125'.bloom.gz') \
	  -k 135 --input-bloom=<(gunzip -c $bloom_path/k135'.bloom.gz') \
	  -k 145 --input-bloom=<(gunzip -c $bloom_path/k145'.bloom.gz') \
	  -k 155 --input-bloom=<(gunzip -c $bloom_path/k155'.bloom.gz') \
	  -k 165 --input-bloom=<(gunzip -c $bloom_path/k165'.bloom.gz') \
	  -k 175 --input-bloom=<(gunzip -c $bloom_path/k175'.bloom.gz') \
	  -k 185 --input-bloom=<(gunzip -c $bloom_path/k185'.bloom.gz') \
	  -k 195 --input-bloom=<(gunzip -c $bloom_path/k195'.bloom.gz') \
	  -k 205 --input-bloom=<(gunzip -c $bloom_path/k205'.bloom.gz') 
done

cd $unfixed_dir
for directory in */; do
  gap_id=$(echo $directory | sed 's/\///g')
  bloom_path=${curr_dir}/sealer_out/unfixed/$gap_id
  out_path=$bloom_path/sealed
  
  mkdir -p $out_path
  
  gap_fasta=$unfixed_dir/$gap_id/gap_draft/$gap_id'_'hg38_aln_full_gap.fasta
  abyss-sealer -v -S $gap_fasta \
	  -t $out_path/$gap_id-sealed-trace.txt \
	  -o $out_path/$gap_id'.sealer' \
	  -L 150 \
	  -j 48 \
	  -P 10  \
	  -k 75 --input-bloom=<(gunzip -c $bloom_path/k75'.bloom.gz') \
	  -k 85 --input-bloom=<(gunzip -c $bloom_path/k85'.bloom.gz') \
	  -k 95 --input-bloom=<(gunzip -c $bloom_path/k95'.bloom.gz') \
	  -k 105 --input-bloom=<(gunzip -c $bloom_path/k105'.bloom.gz') \
	  -k 115 --input-bloom=<(gunzip -c $bloom_path/k115'.bloom.gz') \
	  -k 125 --input-bloom=<(gunzip -c $bloom_path/k125'.bloom.gz') \
	  -k 135 --input-bloom=<(gunzip -c $bloom_path/k135'.bloom.gz') \
	  -k 145 --input-bloom=<(gunzip -c $bloom_path/k145'.bloom.gz') \
	  -k 155 --input-bloom=<(gunzip -c $bloom_path/k155'.bloom.gz') \
	  -k 165 --input-bloom=<(gunzip -c $bloom_path/k165'.bloom.gz') \
	  -k 175 --input-bloom=<(gunzip -c $bloom_path/k175'.bloom.gz') \
	  -k 185 --input-bloom=<(gunzip -c $bloom_path/k185'.bloom.gz') \
	  -k 195 --input-bloom=<(gunzip -c $bloom_path/k195'.bloom.gz') \
	  -k 205 --input-bloom=<(gunzip -c $bloom_path/k205'.bloom.gz') 
done

python3.6 ${curr_dir}/extract_sealer_gaps.py ${curr_dir}/sealer_out

echo "Job ended at $(date)"

### EOF ###
