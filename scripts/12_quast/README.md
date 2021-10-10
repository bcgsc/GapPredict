# QUAST Assessment

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6
* QUAST 5.0.2 ([link](http://bioinf.spbau.ru/quast))

Also, ensure you complete all previous steps, as this step aggregates all data.

## Overview
This step of our pipeline first pools all gap sequences from the reference into a single FASTA file. Then, it pools all predicted gaps from GapPredict, GAPPadder, and Sealer into a separate FASTA file for each tool. Finally, it runs QUAST to evaluate each tools' outputs with respect to the reference gaps. 

Note that in our pipeline, we specifically only compare a subset of gaps. Namely, set 1 gaps that are commonly filled between GapPredict, GAPPadder, and Sealer. By picking this subset, we are focusing on gaps that all three tools would have filled and have been accepted (as set 2 gaps would be rejected since they represent gaps where the reciprocal flank does not confidently align).

## Usage
Each of the four gap pooling scripts follow a similar usage pattern:

`./<pool>.sh <output path> <gap id path> <set 1 gap prediction path>`
* The output path is where you want the FASTA file of all pooled gaps to go
* The gap ID path is the path to a text file where each line contains the ID of a gap to fetch
  * We used the CSVs parsed from Exonerate results in order to obtain the necessary set 1 gaps
* The set 1 gap prediction path is the path to where each tool output its results:
  * For the reference, this is the `fixed` directory from [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference)
  * For GapPredict, this is the `validation/fixed` directory from [step 7] (https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/7_gappredict_local_alignment)
  * For Sealer, this is the `sealer_out/fixed` directory from [step 8](https://github.com/bcgsc/GapPredict/tree/v1.0doc/scripts/8_sealer_gap_filling_in_isolation)
  * For GAPPadder, this is the `align/fixed` directory from [step 11](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/11_gappadder_local_alignment)

Finally run the following:

`./quast.sh <quast path> <output path> <pooled gaps path>`
* The QUAST path is where your QUAST Python executable is located for Python to run
* The output path is the folder where you want your QUAST results output
* The pooled gaps path is the output path from the previous `<pool>.sh` scripts

## Output
The first step produces a FASTA file for each tool containing all the relevant gaps pooled together.

The second step takes the FASTA file for each tool and uses QUAST to compare against the pooled sequences from the reference. It outputs everything that QUAST outputs for such a comparison
