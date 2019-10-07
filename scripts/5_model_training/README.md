# Model Training

## Pre-requisites
The following software should be installed and added to your PATH
* GapPredict v1.0b ([link](https://github.com/bcgsc/GapPredict/releases/tag/v1.0b)) (and all of its dependencies)

Also, ensure you complete [step 4](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/4_random_gap_sampling).

## Overview
This step trains a GapPredict model for each of the gaps sampled from the previous step. Note that this takes at least 30 minutes per gap. 

The provided script trains a GapPredict model for a single gap. Based on your directory structure, the script should be modified to iterate over
every gap. In addition, if multiple GPUs are available, you can partition the set of gaps over all the GPUs.

## Usage
`./train_gap.sh <GapPredict lib directory> <FASTA for a single gap> <FASTQ for a single gap> <output directory>`
* the `lib` directory is the location of [this directory](https://github.com/bcgsc/GapPredict/tree/master/lib) from the GapPredict repository
* each gap sampled from the previous step has an ID (eg. 123_12-32), for which a directory has been made
** the FASTA input for this script is the `123_12-32.fasta` file
** the FASTQ input for this script is the `123_12-32.fastq` file
* the output directory is where you want your trained model artifacts to be stored

The default parameters for GapPredict are the same parameters that we used during our analyses, so only the input FASTA, input FASTQ, and output directory need to be supplied
If you're training independent partitions using multiple GPUs, add a `-gpu <gpu number>` option to the `GapPredict.py` call.

Ensure that predictions for gaps which Sealer could not fill all go in one directory, and predictions for gaps which Sealer could fill all go in another directory.
