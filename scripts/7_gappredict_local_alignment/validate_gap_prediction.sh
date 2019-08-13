#!/bin/bash
if [ $# -lt 3 ]; then
	echo "Usage: $(basename $0) <flanks and gaps directory> <directory of models for gaps filled by Sealer> <directory of models for gaps not filled by Sealer>"
	exit 1
fi

flanks_and_gaps=$1;shift
fixed_dir=$1;shift
unfixed_dir=$1;shift

current_dir=$(pwd)

fix=fixed
valid_dir=${current_dir}/validation/$fix
mkdir -p $valid_dir
all_models=${fixed_dir}
cd $all_models
for model in */; do
  gap_id=$(echo $model)
  
  reference_path=${flanks_and_gaps}/$fix/$gap_id
  left_flank_fasta=$reference_path/flanks/$gap_id'_'left_flank.fasta
  right_flank_fasta=$reference_path/flanks/$gap_id'_'right_flank.fasta
  gap_fasta=$reference_path/gap/$gap_id'_'hg38_aln_gap.fasta
  subflanks_out_path=$valid_dir/$gap_id/flanks

  mkdir -p $subflanks_out_path
  python3.6 ${current_dir}/extract_subflanks.py -lf $left_flank_fasta -rf $right_flank_fasta -o $subflanks_out_path -l 100

  inner_path=beam_search/predict_gap
  model_path=$all_models/$gap_id/$gap_id'_R_0'
  predictions_out_path=$valid_dir/$gap_id/predictions
  predictions_path=$model_path/$inner_path
  forward=forward
  rc=reverse_complement
  beam_search_predict_file=predict.fasta
  top_one_fwd=top_one_forward_predict.fasta
  top_one_rc=top_one_rc_predict.fasta

  mkdir -p $predictions_out_path

  head -2 $predictions_path/$forward/$beam_search_predict_file > $predictions_out_path/$top_one_fwd
  head -2 $predictions_path/$rc/$beam_search_predict_file > $predictions_out_path/$top_one_rc
  python3.6 ${current_dir}/process_predictions.py -lf $predictions_out_path/$top_one_fwd -rf $predictions_out_path/$top_one_rc

  exonerate_out_path=$valid_dir/$gap_id/exonerate

  mkdir -p $exonerate_out_path
  left_subflank_path=$subflanks_out_path/left_subflank.fasta
  right_subflank_path=$subflanks_out_path/right_subflank.fasta
  exonerate $predictions_out_path/$top_one_rc $left_subflank_path --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $exonerate_out_path/left_subflank_right_prediction.exn
  exonerate $predictions_out_path/$top_one_fwd $right_subflank_path --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/right_subflank_left_prediction.exn
  exonerate $predictions_out_path/$top_one_rc $gap_fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/gap_right_prediction.exn
  exonerate $predictions_out_path/$top_one_fwd $gap_fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/gap_left_prediction.exn
done

fix=unfixed
valid_dir=${current_dir}/validation/$fix
mkdir -p $valid_dir
all_models=${unfixed_dir}
cd $all_models
for model in */; do
  gap_id=$(echo $model)
  
  reference_path=${flanks_and_gaps}/$fix/$gap_id
  left_flank_fasta=$reference_path/flanks/$gap_id'_'left_flank.fasta
  right_flank_fasta=$reference_path/flanks/$gap_id'_'right_flank.fasta
  gap_fasta=$reference_path/gap/$gap_id'_'hg38_aln_gap.fasta
  subflanks_out_path=$valid_dir/$gap_id/flanks

  mkdir -p $subflanks_out_path
  python3.6 ${current_dir}/extract_subflanks.py -lf $left_flank_fasta -rf $right_flank_fasta -o $subflanks_out_path -l 100

  inner_path=beam_search/predict_gap
  model_path=$all_models/$gap_id/$gap_id'_R_0'
  predictions_out_path=$valid_dir/$gap_id/predictions
  predictions_path=$model_path/$inner_path
  forward=forward
  rc=reverse_complement
  beam_search_predict_file=predict.fasta
  top_one_fwd=top_one_forward_predict.fasta
  top_one_rc=top_one_rc_predict.fasta

  mkdir -p $predictions_out_path

  head -2 $predictions_path/$forward/$beam_search_predict_file > $predictions_out_path/$top_one_fwd
  head -2 $predictions_path/$rc/$beam_search_predict_file > $predictions_out_path/$top_one_rc
  python3.6 ${current_dir}/process_predictions.py -lf $predictions_out_path/$top_one_fwd -rf $predictions_out_path/$top_one_rc

  exonerate_out_path=$valid_dir/$gap_id/exonerate

  mkdir -p $exonerate_out_path
  left_subflank_path=$subflanks_out_path/left_subflank.fasta
  right_subflank_path=$subflanks_out_path/right_subflank.fasta
  exonerate $predictions_out_path/$top_one_rc $left_subflank_path --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y --bestn 1 > $exonerate_out_path/left_subflank_right_prediction.exn
  exonerate $predictions_out_path/$top_one_fwd $right_subflank_path --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/right_subflank_left_prediction.exn
  exonerate $predictions_out_path/$top_one_rc $gap_fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/gap_right_prediction.exn
  exonerate $predictions_out_path/$top_one_fwd $gap_fasta --ryo "percent_identity:%pi\nquery_start:%qab\nquery_length:%ql\nquery_alignment_length:%qal\nquery_end:%qae\ntarget_start:%tab\ntarget_end:%tae\ntarget_length:%tl\ntarget_alignment_length:%tal\ntotal_bases_compared:%et\nmismatches:%em\nmatches:%ei\ncigar:%C\n" --model affine:local --exhaustive y  --bestn 1 > $exonerate_out_path/gap_left_prediction.exn
done

csv_dir=${current_dir}/csv
mkdir -p $csv_dir
python3.6 ${current_dir}/aggregate_exonerate.py ${current_dir}/validation ${csv_dir}