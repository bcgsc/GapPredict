#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <fixed reads dir> <unfixed reads dir>"
	exit 1
fi
mkdir -p logs

fix_dir=$1;shift
unfix_dir=$1;shift
curr_dir=$(pwd)
for i in {75..205..10}
  do
    nohup ./runme_ABySS-bloom.sh $i $fix_dir $unfix_dir $curr_dir &> logs/k$i.log &
  done
