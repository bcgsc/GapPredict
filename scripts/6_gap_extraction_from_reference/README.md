# Reference Gap Acquisition

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6
* Hg38 reference genome FASTA ([link](https://www.ncbi.nlm.nih.gov/assembly?term=GRCh38&cmd=DetailsSearch))
* BWA 0.7.17 ([link](https://github.com/lh3/bwa/releases/tag/v0.7.17))
* BEDtools 2.27.1 ([link](https://github.com/arq5x/bedtools2/releases/tag/v2.27.1))
* SAMtools 1.9 ([link](https://github.com/samtools/samtools/releases/tag/1.9))

Also, ensure you complete [step 4](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/4_random_gap_sampling). It's recommended you do this step while your models are being trained in [step 5](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training).

## Overview
We make a copy of all the flank FASTA files first.

This script then obtains sequences for each sampled gap from the HG38 human reference genome.

In order to do this, consider any gap. We only have data on the flanking sequences.

We first use `bwa mem` to align the flanks of our gap to HG38. Then, we convert the BAM file into a BED file (`bedtools bamtobed`) and extract the coordinates
of where the flanks align in HG38. `bedtools merge` is then used to obtain the full (left flank)-gap-(right flank) sequence. We then run a Python script to
obtain only the gap sequence itself (without the flanks). This Python script also gives us an opportunity to filter out even more gaps - the ones whose BED files
are not well formatted, and hence will have strange looking reference sequences.

Finally, we use `bedtools getfasta` to convert the BED file coordinates to actual sequences from the reference.

We repeat this process (without the Python script filtering) on the draft assembly in order to get the flanks and gap prior to gap filling (so the gap has N's).

## Usage
First run `./copy_fastas.sh <fixed gap flank FASTA path> <unfixed gap flank FASTA path>`
* the fixed gap flank FASTA path is the `fixed` directory output from [step 3](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction) of our pipeline
* the unfixed gap flank FASTA path is the `unfixed` directory output from [step 3](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction) of our pipeline

Then, run `./gap_extract_pipeline.sh <HG38 path> <draft assembly path>`
* the HG38 path is the path to the HG38 FASTA file, which should be downloaded
* the draft assembly path is the path to the `human-scaffolds.fa` file from [step 1](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/1_abyss_assembly)

## Output
We create a `fixed` and `unfixed` directory in the current directory containing directories for each gap copied over. For a given gap, the contents of its directory include:
* FASTA files
  * a `_flanks.fasta` file with both flanks
  * a `_left_flank.fasta` file with only the left flank
  * a `_right_flank.fasta` file with only the right flank
  * a `_full_gap.fasta` file with the full gap sequence from HG38 with flanks
  * a `_gap.fasta` file with only the gap sequence from HG38
* BAM files for the alignment to HG38
* BED files 
  * a `_flanks.bed` file with the coordinates of each flank separately in HG38
  * a `_merged_flanks.bed` file with the coordinates of a the (left flank)-gap-(right flank) sequence in HG38
  * a `_gaps.bed` file with the coordinates of the gap only in HG38
* a `dirty` directory where erroneous gaps have been moved to

And analogous for the draft genome files. You shouldn't need to worry about the contents of these directories as we assume the directory structure is preserved.
