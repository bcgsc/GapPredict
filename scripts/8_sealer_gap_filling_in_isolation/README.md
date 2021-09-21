# Sealer Re-Run Using Reads Isolated for a Gap

## Pre-requisites
The following software should be installed and added to your PATH
* ABySS 2.1.5 ([link](https://github.com/bcgsc/abyss/releases/tag/2.1.5))
  * based on the k-mer length parameter we used, a max-k configuration of 256 is suggested
* Python 3.6

Also, ensure you complete [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference). You do not need to have [step 5](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training) completed.

## Overview
This step makes Sealer try to fill all of the sampled gaps (both the ones that Sealer was able to fill and unable to fill back in step 2) again using Bloom
filters ranging from 75-mers to 205-mers. The difference with this iteration of running Sealer is that for each gap, each Bloom filter used by Sealer to fill
the gap is made from reads that map only to the gap. Compare this with step 2) where for each gap, each Bloom filter used by Sealer to fill the gap is made
from all reads in the NA12878 dataset.

We do this because GapPredict doesn't train on all the reads in NA12878 to fill the gap, so we'll make Sealer use only the reads that GapPredict would have used
to fill the gap. It's expected that Sealer's gap filling performance won't change that much. All of the gaps which Sealer was unable to fill in step 2) should remain
unfilled and all of the gaps which Sealer was able to fill in step 2) should remain filled... give or take a few gaps.

After running Sealer, we run a Python script on Sealer's output to get only the sequence Sealer provided for the gap itself, throwing away the flanks. This is only run on
gaps which Sealer actually produced a `.sealer_merged` file for that isn't blank.

## Usage
First run `./run_multik_bloom.sh <fixed reads dir> <unfixed reads dir>`
* the fixed reads directory is the `fixed` directory from [step 4](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/4_random_gap_sampling) of the pipeline containing directories for each of the gaps Sealer was able to fill
* the unfixed reads directory is the `unfixed` directory from [step 4](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/4_random_gap_sampling) of the pipeline containing directories for each of the gaps Sealer was unable to fill

These input directories hold read data for each gap individually.

Next run `./run-sealerAllReads.sh <fixed flanks and gaps dir> <unfixed flanks and gaps dir>`
* the fixed flanks and gaps directory is the `fixed` directory from [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference) of the pipeline containing directories for each of the gaps Sealer was able to fill
* the unfixed flanks and gaps directory is the `unfixed` directory from [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference) of the pipeline containing directories for each of the gaps Sealer was unable to fill

These input directories contain the flank-gap-flank construct prior to gap filling.

## Output
In the current directory we output a `sealer_out` directory. This contains a `fixed` and `unfixed` directory, for gaps filled by Sealer in step 2 and gaps not filled
by Sealer in step 2 respectively. Both `fixed` and `unfixed` contain a directory for each gap in which you'll find the Bloom filters for the gap.

We also output a `sealed` directory inside each gap's directory. This contains the filled in gap, if Sealer was able to fill it in, in addition to all the other Sealer output.
