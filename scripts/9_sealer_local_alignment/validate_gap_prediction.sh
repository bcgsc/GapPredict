#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <sealer output dir> <flanks and gaps dir>"
	exit 1
fi

sealer_out=$1;shift
flanks_and_gaps=$1;shift

curr_dir=$(pwd)

fix=fixed
valid_dir=${curr_dir}/validation/$fix
mkdir -p $valid_dir

all_models=${sealer_out}/$fix
reference_path=${flanks_and_gaps}/$fix
cd $all_models
for model in */; do
  gap_id=$(echo $model)
  sealer_filled=$all_models/$gap_id/sealed/$gap_id'_gap.fasta'
  sealer_merged=$all_models/$gap_id/sealed/$gap_id'.sealer_merged.fa'
  gap_reference=$reference_path/$gap_id/gap/$gap_id'_'hg38_aln_gap.fasta
  if [ -f $sealer_filled ]
  then 
    out_path=$valid_dir/$gap_id
    mkdir -p $out_path
    
    exonerate $sealer_filled $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer.exn
    exonerate $sealer_merged $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer_merged.exn
  fi  
done

fix=unfixed
valid_dir=${curr_dir}/validation/$fix
mkdir -p $valid_dir

all_models=${sealer_out}/$fix
reference_path=${flanks_and_gaps}/$fix
cd $all_models
for model in */; do
  gap_id=$(echo $model)
  sealer_filled=$all_models/$gap_id/sealed/$gap_id'_gap.fasta'
  sealer_merged=$all_models/$gap_id/sealed/$gap_id'.sealer_merged.fa'
  gap_reference=$reference_path/$gap_id/gap/$gap_id'_'hg38_aln_gap.fasta
  if [ -f $sealer_filled ]
  then 
    out_path=$valid_dir/$gap_id
    mkdir -p $out_path
    
    exonerate $sealer_filled $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer.exn
    exonerate $sealer_merged $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer_merged.exn
  fi  
done

python3.6 ${curr_dir}/aggregate_exonerate.py ${curr_dir}/validation/ ${curr_dir}
