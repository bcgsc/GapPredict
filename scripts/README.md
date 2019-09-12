# GapPredict Pipeline Scripts

This directory contains the scripts needed to reproduce our complete pipeline for GapPredict, in which we assemble a draft genome for the NA12878 human genome, fill its gaps with Sealer, fill its gaps with GapPredict, and assess how well each tool performed. The general idea of this pipeline should be applicable to any genome assembly.

We've divided our pipeline into 9 steps:

1) ABySS Draft Genome Assembly ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/1_abyss_assembly))
2) Sealer gap filling ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/2_sealer_gap_filling))
3) Identifying gaps, obtaining the sequences of their flanks, and finding reads mapping to these flanks ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/3_flank_extraction))
4) Sampling gaps which Sealer filled and gaps which Sealer could not fill ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/4_random_gap_sampling))
5) Training GapPredict models for each sampled gap and predicting each gap's sequence ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training))
6) Finding the actual sequence of each sampled gap in HG38 ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/6_gap_extraction_from_reference))
7) Aligning GapPredict's predictions to the HG38 reference sequence for each gap (our source of truth) ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/7_gappredict_local_alignment))
8) Running Sealer on each gap using reads mapping only to a given gap's flanks ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/8_sealer_gap_filling_in_isolation))
9) Aligning Sealer's output from step 8) to the HG38 reference sequence for each gap (our source of truth) ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/9_sealer_local_alignment))

## Pipeline Diagram
**TODO: add this here**

## Caveats
Several scripts are likely not runnable out-of-the-box depending on how your PATH is set up. We also went through some refactoring of our scripts to better automate the process - ideally nothing broke as a result.

Step 5) ([link](https://github.com/bcgsc/GapPredict/tree/Reproduction_Steps/scripts/5_model_training)) will almost certainly need some manual editing of the scripts depending on your computing system's resources.

We decided to make 9 steps in the pipeline to give checkpoints and better isolate any errors that may occur.
