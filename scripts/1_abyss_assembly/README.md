# ABySS Assembly

## Pre-requisites
The following software should be installed and added to your PATH
* ABySS 2.1.5 ([link](https://github.com/bcgsc/abyss/releases/tag/2.1.5))
  * based on the k-mer length parameter we used, a max-k configuration of 160 is suggested
* ksh 5.4.2_3

## Overview
This step uses ABySS to assemble the NA12878 human genome. The reads were obtained from https://basespace.illumina.com (flowcell H00DDBCXX).

It is assumed that the NA12878 directory contains 2 sets of paired end reads, with each set containing two compressed FASTQ files. 
In addition, we ran this script using a high performance computing cluster. If either of these assumptions are not satisfied, the 
`runABySS-bf.sh` script may need to be modified.

## Usage
We ran this script using `./runABySS-bf.sh 144 18 human <NA12878 reads directory> <output directory>`. 

The third parameter must be "human" as subsequent scripts assume this prefix was used.

## Output
ABySS produces the draft assembly in the output directory. Most files can be ignored. The only file of interest for our pipeline is `human-scaffolds.fa`, which contains the actual draft assembly sequences.
