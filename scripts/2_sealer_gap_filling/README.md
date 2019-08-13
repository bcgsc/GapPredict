# Sealer Gap Filling
## Pre-requisites
The following software should be installed and added to your PATH
* ABySS 2.1.5 ([link](https://github.com/bcgsc/abyss/releases/tag/2.1.5))
  * based on the k-mer length parameter we used, a max-k configuration of 256 is suggested
* Python 3.6

Also, ensure you complete [step 1](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/1_abyss_assembly).

## Overview
In order to determine which gaps may be "easier" to fill, we use Sealer to fill as many gaps in our draft assembly from step 1 of our pipeline as possible.

First, we use ABySS-bloom to create Bloom filters for all the reads in NA12878. We create Bloom filters starting from a k-mer length of 75
up to a k-mer length of 205 using an interval of 10 in between each k-mer (ie. 75, 85, ..., 205).

Next, we use these Bloom filters and Sealer to fill as many gaps in our draft assembly (from the previous step) as possible.

Sealer outputs several files but we only focus on the `.sealer_merged` file since this contains the filled in gaps

## Usage
First run `./runme_ABySS-bloom.sh <NA12878 reads directory> <output directory>` 
* the NA12878 reads directory is where the reads for NA12878 data are
* the output directory is where you want your Bloom filters to be saved

Next, run `./run-sealerAllReads.sh <Draft scaffolds FASTA> <Bloom filter directory>` to execute Sealer
* the draft scaffolds FASTA is the FASTA file output by ABySS; it should be called `human-scaffolds.fa`
* the Bloom filter directory is the directory where all the Bloom filters from the previous script are output

## Output
* Bloom filters for k = 75, 85, ..., 205 are output in the specified directory
* Sealer's output will be in the Bloom filter directory, only the `.sealer_merged` file is of particular interest later on
