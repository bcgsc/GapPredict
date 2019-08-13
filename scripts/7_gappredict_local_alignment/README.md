# GapPredict Prediction Alignment

## Pre-requisites
The following software should be installed and added to your PATH
* Python 3.6
* Exonerate 2.2.0 ([link](https://www.ebi.ac.uk/about/vertebrate-genomics/software/exonerate))

Also, ensure you complete [step 5](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training) and [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference).

## Overview
This step of our pipeline uses Exonerate to align GapPredict's prediction for each gap to the HG38 reference sequence for the gap. We then parse Exonerate's output for all gaps and
organize the data in a CSV in order to determine how well GapPredict predicted its gaps.

First we run a Python script to extract the 100 bp from both the right flank and the left flank that is closest to the gap. The left and right flank sequences are taken from the HG38 reference. This is because we expect predictions from the left flank which respectably contain
these 100 bp from the right flank to predict the gap well. The same reasoning applies for predictions from the right flank. We'll refer to the 100 bp from the right flank as the "right subflank" and analogous for the left flank.

Next, we run a Python script to strip off the seed sequence for GapPredict's predictions. Each gap has two predictions - one starting from the right flank as a seed, and one starting from the left flank as a seed. GapPredict
(based on our configurations) then predicts the next 750 bases given the seed. The Python script isolates only the 750 bases predicted by GapPredict by removing the seed. We'll refer to the 750 bases predicted using the
left flank as a seed as the "left prediction", and the 750 bases predicted using the right flank as a seed as the "right prediction".

Finally, we use Exonerate to perform 4 alignments:
a) reference gap sequence to the left prediction
b) reference gap sequence to the right prediction
c) right subflank to the left prediction
d) left subflank to the right prediction

Exonerate performs an exhaustive local alignment and chooses the single best alignment.

Alignments a) and b) are done to see whether the gap sequence was actually predicted. Alignments c) and d) are done to see whether the subflanks are actually predicted by GapPredict. During evaluation of predictions,
we would only know what the subflank sequences are (as the gap itself is unknown), so this alignment gives us a chance to determine if the prediction of the subflank corresponds to accurate gap predictions.

This process is repeated for every gap.

Finally, we run a Python script to iterate over all Exonerate alignments, parse them, and store their data into a CSV. This is used later on to calculate the four metrics (target percent coverage,
query percent coverage, target percent correctness, query percent correctness) mentioned in our paper.

## Usage
Run `./validate_gap_prediction <flanks and gaps directory> <directory of models for gaps filled by Sealer> <directory of models for gaps not filled by Sealer>`
* the flanks and gaps directory is the directory containing the HG38 reference sequences for each gap output by [step 6](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference)
* the latter two arguments were defined by the user in [step 5](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training)

## Output
Outputs into this directory.

For each gap:
* a `flanks` directory with flank sequences and subflank sequences
* a `gaps` directory with GapPredict's predictions, without the seed
* a `exonerate` directory with Exonerate's alignment output

In addition, a `csv` directory is created containing one CSV for each of alignments a)-d). Each CSV contains data for every gap. You can wrangle and visualize these CSVs by modifying the provided [script](https://github.com/bcgsc/GapPredict/blob/Reproduction_Steps/lib/data/helper_scripts/data_crunching/wrangle_csv.py).
