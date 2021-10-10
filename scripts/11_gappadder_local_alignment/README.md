# GAPPadder Prediction Alignment

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6
* Exonerate 2.2.0 ([link](https://www.ebi.ac.uk/about/vertebrate-genomics/software/exonerate))
* BWA 0.7.17 ([link](https://github.com/lh3/bwa/releases/tag/v0.7.17))
* SAMtools 1.9 ([link](https://github.com/samtools/samtools/releases/tag/1.9))

Also, ensure you complete [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference) and [step 10](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/10_gappadder_gap_filling).

## Overview
This step of our pipeline uses BWA-mem to determine where each gap aligns to in GAPPadder's output, if it aligns at all. It then uses SAMtools to fetch the filled gap sequence from GAPPadder and finally uses Exonerate to align GAPPadder's output to the HG38 reference sequence for the gap. 

Finally, an in-house Python script is used to aggregate the Exonerate data into a CSV with consistent metrics to Sealer and GapPredict.

This step is not much different in essence than Steps 7 and 9.

## Usage
Run `./prepare.sh <picked_seqs.fa_ori.txt>` to index GAPPadder's output.
* the `picked_seqs.fa_ori.txt` file can be found where GAPPadder wrote its output
  * Note: BWA will output its index wherever the `picked_seqs.fa` file is located. It may be a good idea to copy it to its own directory to be tidy.
  * Note: Do not use `picked_seqs.fa`, this doesn't include any data about the gap flanks. Feel free to rename the fa_ori.txt file to something with a `.fa` extension 
 
Run `./align.sh <flanks and gaps directory> <indexed picked_seqs.fa_ori.txt> <out dir>` to perform the necessary alignments and run Exonerate.
* the flanks and gaps directory is the directory containing the `fixed` and `unfixed` directories from [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference) of the pipeline.
* the `indexed picked_seqs.fa_ori.txt` is from running `prepare.sh` earlier in this step
* the `out` directory is the folder where you'd like all the outputs of this step to go to

Finally, run `python3.6 ./aggregate_exonerate_gappadder.py <out dir>/align <csv output directory>` to aggregate the Exonerate results.
* the `out` directory is the folder that `align.sh` output its results to 
* the CSV output directory is a directory of your choice to output the CSV file to

## Output
Creates an index and all its artifacts where the `picked_seqs.fa_ori.txt` file is.

Also, in the `out` directory, a folder will be made for every gap where artifacts of this step - BAM files, FASTA files with gap sequences, and Exonerate files - are output.

The Python script will produce a `gappadder.csv` file with all the aggregated metrics from Exonerate.