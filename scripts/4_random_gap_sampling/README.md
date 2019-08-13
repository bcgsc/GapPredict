# Gap Sampling

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6

Also, ensure you complete [step 3](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction).

## Overview
This step of the pipeline splits all the gaps found in the [previous step](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction) into gaps that either Sealer was able to fill or Sealer was unable to fill.
It then randomly samples some number of these gaps and obtains both the flanking sequences of these gaps and the reads mapping to these flanks.

GapPredict has some limitations so we have to do some filtering steps to remove sampled gaps which are more complicated to handle. In particular, these are gaps that:
* are shorter than 20 bp
* have flanks shorter than 500 bp
* have flanks containing "N" or "n" (hence are flanks with gaps inside them)

## Usage
Run `./sample_random_gaps.sh <gaps file> <sealer merged file> <output path> <sample size> <reads file>`
* the gaps file is the `human-scaffolds_gaps_flanks.fa` file from the [previous step](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction)
* the `.sealer_merged` file is one of the outputs from [running Sealer](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/2_sealer_gap_filling)
* the output path is where you want this step's output to go to
* sample size refers to how many gaps you want to randomly sample (we used 900, for example)
* the reads file is the compressed FASTQ of all reads mapping to flanks from the [previous step](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction)

## Output
The primary output of this step is an `unfixed` and `fixed` directory in the output path. The `fixed` directory is for gaps which Sealer managed to fill
and the `unfixed` directory for gaps which Sealer could not fill. In addition, the output path will also contain a `dirty` directory with the gaps that were
filtered out due to failing to meet the criteria specified above.

Within these directories is a folder for each sampled gap which passed the filter containing
a FASTA file with the flanks for the gap, and a FASTQ file with the reads mapping to these flanks. 
