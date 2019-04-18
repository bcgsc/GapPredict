#!/bin/bash
PATH=/gsc/btl/linuxbrew/bin/:$PATH

fix=$1;shift

valid_dir=./validation/$fix
mkdir -p $valid_dir

all_models= sealer prediction output
reference_path=path containing gap
for model in $all_models/*/; do
  gap_id=$(echo $model | cut -f9 -d '/')
  sealer_filled=$all_models/$gap_id/sealed/$gap_id'_gap.fasta'
  sealer_merged=$all_models/$gap_id/sealed/$gap_id'.sealer_merged.fa'
  gap_reference=$reference_path/$gap_id/gap/$gap_id'_'hg38_aln_gap.fasta
  if [ -f $sealer_filled ]
  then 
    out_path=$valid_dir/$gap_id
    mkdir -p $out_path
    
    exonerate $sealer_filled $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer.exn
    exonerate $sealer_merged $gap_reference --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\n" --model affine:local --exhaustive y --bestn 1 > $out_path/sealer_merged.exn
  fi  
done
