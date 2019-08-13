# GapPredict Prediction Alignment

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6
* Exonerate 2.2.0 ([https://www.ebi.ac.uk/about/vertebrate-genomics/software/exonerate](link))

Also, ensure you complete [step 8](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/8_sealer_gap_filling_in_isolation). You do not need to have [step 5](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training) completed.

## Overview
This step of our pipeline uses Exonerate to align Sealer's output for each gap to the HG38 reference sequence for the gap. We then parse Exonerate's output for all gaps and
organize the data in a CSV in order to determine how well Sealer filled its gaps.

We use Exonerate to perform 2 alignments:
a) one alignment is between the full `.sealer_merged` file and the HG38 reference
b) the other alignment is between the filled gap without the flanks, and the HG38 reference

Exonerate performs an exhaustive local alignment and chooses the single best alignment.

This process is repeated for every gap.

Ultimately we didn't actually use alignment b) during our metrics because the flank removal process was buggy.

Finally, we run a Python script to iterate over all Exonerate alignments, parse them, and store their data into a CSV. This is used later on to calculate the target percent correctness. 

## Usage
Run `./validate_gap_prediction <sealer output dir> <flanks and gaps dir>`
* the Sealer output directory is the `sealer_out` directory from [step 8](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/8_sealer_gap_filling_in_isolation).
* the flanks and gaps directory is the directory containing the `fixed` and `unfixed` directories from [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference) of the pipeline.

## Output
Creates a `validation` directory with all the Exonerate output.

In addition, CSVs for alignments a) and b) are output into this directory. Each CSV contains data for every gap. You can wrangle and visualize these CSVs by modifying the provided [script](https://github.com/bcgsc/GapPredict/blob/Reproduction_Steps/lib/data/helper_scripts/data_crunching/wrangle_sealer.py).
