# GAPPadder Gap Filling

## Pre-requisites
The following software should be installed and added to your PATH
* GAPPadder ([link](https://github.com/simoncchu/GAPPadder))
* BWA 0.7.17 ([link](https://github.com/lh3/bwa/releases/tag/v0.7.17))
* SAMtools 1.9 ([link](https://github.com/samtools/samtools/releases/tag/1.9))

Also, ensure you complete [step 1](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/1_abyss_assembly).

## Overview
This step of our pipeline fills all gaps in from the ABySS scaffold using GAPPadder, as an additional comparison for GapPredict's performance. First, we use BWA-mem to align NA12878 reads to the scaffold and obtain some statistics. These are then used in GAPPadder's configuration.json so that we can run GAPPadder properly.

## Usage
Run `./prepare_alignments.sh <alignment output path> <scaffold file> <reads path>`
* the alignment output path is where you'd like BWA to output its alignments and SAMtools to output its statistics
* the scaffold file is path to the `human-scaffolds.fa` file from Step 1
* the reads path is the path to the reads directory used in Step 1
  * note that we assume the file names of the reads files, similar to Step 1

Next, GAPPadder's `configuration.json` needs to be filled out. We have provided a template where only file paths and the `is` and `std` statistics for the `alignment` property need to be filled out. 
* `is` refers to the `Insert Size Average` metric from `samtools stats`
* `std` refers to the `Insert Size Standard Deviation` metric from `samtools stats`

Once this is done, you can run `./gappadder.sh <GAPPadder repo path> <GAPPadder output path>`
* the GAPPadder repo path is just the path to the GAPPadder folder
* the GAPPadder output path is where you want GAPPadder to write its output to, so the shell script can write its logs

Note: We ran into some errors when we used GAPPadder. [Here](https://github.com/EricChen424/GAPPadder) is a link to a forked repository that should unblock any errors during runtime. Changes are listed [here](https://github.com/EricChen424/GAPPadder/pull/1/files).

## Output
Creates a `bwa` directory with all alignments and alignment stats.

In addition, GAPPadder will create its output where specified in the `configuration.json`.