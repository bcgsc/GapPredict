# Gap Read Extraction

## Pre-requisites
The following software should be installed and added to your PATH
* Perl 5
* Python 3.6
* BEDtools 2.27.1 ([link](https://github.com/arq5x/bedtools2/releases/tag/v2.27.1))
* SAMtools 1.9 ([link](https://github.com/samtools/samtools/releases/tag/1.9))
* BioBloom Tools 2.3.2 ([link](https://github.com/bcgsc/biobloom/releases/tag/2.3.2))

Also, ensure you complete [step 1](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/1_abyss_assembly)

## Overview
This step in the pipeline uses the previously assembled ABySS draft assembly in order to identify all gaps, obtain their flanks, and find reads mapping to these flanks.

First we use a Perl script in order to convert the scaffolds file into an AGP file which we can search for gaps in (via presence of "N"). We store data for these gaps in a BED file.

The scaffold is then indexed by SAMtools and we use this index, our previous BED file, and BEDtools `flank` to obtain coordinates for the 500 bp left and right flanks for each gap.

BEDtools `getfasta` is then used to get the actual sequences for these flanks.

BioblooMImaker is used to make a multi-indexed Bloom filter for each of the flanks. BioblooMIcategorizor can then use this multi-indexed Bloom filter to iterate over all the NA12878 reads
and extract only reads that likely map to the flanking sequences.

## Usage
Run `./extract_flanks_and_reads.sh <scaffold path> <output directory> <NA12878 reads directory>`

## Output
In the output directory are 2 particular files of interest:
* `human-scaffolds_gaps_flanks.fa` - contains all flank sequences for every gap in the draft assembly from step 1
* `NA12878_mappedSubset.fastq.gz` - contains reads mapping to any of these flanks
