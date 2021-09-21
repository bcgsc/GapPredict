#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage: $(basename $0) <GAPPadder repo path> <GAPPadder output path>"
	exit 1
fi

gappadder_path=$1;shift
gappadder_output=$1;shift

cd $gappadder_path
python ${gappadder_path}/main.py -c Preprocess -g ${gappadder_path}/configuration.json > ${gappadder_output}/preprocess.logs.txt

python ${gappadder_path}/main.py -c Collect -g ${gappadder_path}/configuration.json > ${gappadder_output}/collect.logs.txt

python ${gappadder_path}/main.py -c Assembly -g ${gappadder_path}/configuration.json > ${gappadder_output}/assembly.logs.txt