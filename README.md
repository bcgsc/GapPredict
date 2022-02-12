# GapPredict
## About
GapPredict is an LSTM character-level language model that can be used for gap filling de novo assemblies. GapPredict can predict the bases of a single gap using reads mapping to known flanking sequences of the gap. 

In its current implementation, GapPredict will predict a user-defined number of bases after both the left flank and the right flank of the gap. If the user chooses a sufficiently long prediction length, then GapPredict may be able to predict both the gap and the reciprocal flank. A downstream local alignment tool (eg. Exonerate [1]) can be then used to align, say, the first 100 bases of the reciprocal flank to the prediction. If these 100 bases align with high % identity and % coverage (eg. >90%), then it is likely the preceding bases are a good prediction of the actual gap.

## Installing GapPredict
Please ensure you are using Python3.7. Dependencies were last tested with Python3.7.9. Refer to https://www.tensorflow.org/install/gpu to hook up your GPU if you don't have another set-up in place.
In the event that v1.1 doesn't work, try using v1.0.2 with Python3.6 and file an issue. v1.0.2 contains the intended configuration with Tensorflow 1.5.

You can install GapPredict by cloning or downloading the .zip file directly from GitHub.

`git clone git@github.com:bcgsc/GapPredict.git`

GapPredict uses Python3.6 and packages outlined in requirements.txt (https://github.com/bcgsc/GapPredict/blob/master/requirements.txt). These packages can be quickly installed by running:

`pip install -r requirements.txt`

or

`python -m pip install -r requirements.txt`

In order to train models and predict efficiently, a GPU is mandatory. Steps to install CUDA and cuDNN are available at the following links:

* CUDA: https://docs.nvidia.com/cuda/index.html
* cuDNN: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

## Input Data Preparation
GapPredict requires two input files - a FASTA file containing the sequences of the left and right flanks of your gap, and a FASTQ file containing reads mapping to your gap flanks. The length of the flanks can be arbitrary, however in our tests, we used uniform lengths of 500 bp.

We used BioBloomTools BioBloomMIMaker [2], followed by BioBloomTools BioBloomMICategorizer [2] to obtain reads from the full set of reads used in the de novo assembly that map only to our gap's flanks.

## Gap Prediction With GapPredict
To run GapPredict, navigate to the `lib` directory and call:

`python GapPredict.py -o <output directory> -fa <FASTA path> -fq <FASTQ path>`

For help, call:

`python GapPredict.py --help`

We've provided sample FASTA and FASTQ files in `lib/data/real_gaps/sealer_filled` and `lib/data/real_gaps/sealer_unfilled`. Gaps in `lib/data/real_gaps/sealer_filled` have been filled by Sealer [3], a state-of-the-art gap-filling tool, so we've also included Sealer's output for the actual gap sequence to use as a reference. The human reference genome (HG38) must be used to obtain a reference sequence for gaps in `lib/data/real_gaps/sealer_unfilled` (in addition to gaps in `lib/data/real_gaps/sealer_filled`).

eg. `python GapPredict.py -o <output directory> -fa .../lib/data/real_gaps/sealer_filled/7391826_358-1408.fasta -fq .../lib/data/real_gaps/sealer_filled/7391826_358-1408.fastq`

Where `...` is the absolute path to the `lib` directory.
## GapPredict Outputs
GapPredict outputs the following directory structure:

Root directory (\<gap ID\>_R_\<replicate number\>)
* **beam_search** - contains results from predicting the flanks and gaps using beam search
  * **predict_gap** - contains results from predicting the gap from both the forward and reverse complement direction using beam search with a user specified beam length
    * inner directories specifying which direction the gap was predicted from
      * **beam_search_predicted_probabilities.npy** - vector of length B (beam length) of log-sum probabilities for each predicted gap, in descending order 
      * **beam_search_predictions.fasta** - file of the top B gap predictions for the gap from the given direction
  * **regenerate_seq** - contains results from predicting the left flank and the right flank from both the forward and reverse complement direction using beam search with a user specified beam length
    * inner directories specifying the left/right flank and which direction the flank was predicted from
      * **beam_search_predicted_probabilities.npy** - vector of length B (beam length) of log-sum probabilities for each predicted flank, in descending order 
      * **beam_search_predictions.fasta** - file of the top B gap predictions for the given flank from the given direction
* **predict_gap** - contains results from predicting the gap from both the forward and reverse complement direction using the greedy algorithm (beam search with a beam length of 1)
    * inner directories specifying which direction the gap was predicted from
      * **predicted_probabilities.npy** - P x 4 matrix (where P is the length predicted) containing the probability vector output by the GapPredict model for each predicted base
* **regenerate_seq** - contains results from predicting the left flank and the right flank from both the forward and reverse complement direction using the greedy algorithm (beam search with a beam length of 1)
    * **flank_predict.fasta** - contains the left flank and right flank predicted from both the forward and reverse complement directions (4 sequences total)
    * inner directories specifying the left/right flank and which direction the flank was predicted from
      * **greedy_predicted_probabilities.npy** - contains the log-sum probability for the greedy prediction
      * **predicted_probabilities.npy** - P x 4 matrix (where P is the length predicted) containing the probability vector output by the GapPredict model for each predicted base
      * **random_predicted_probabilities.npy** - vector of length P with the probability for each randomly chosen base
      * **teacher_force_predicted_probabilities.npy** - vector of length P with the probability of each base chosen to match the actual reference sequence
* **BS_\<batch size\>_ED_\<embedding dimensions\>_LD_\<LSTM cells\>_E_\<epochs\>_R_\<replicate\>** - contains model training results
  * contains graphs for training loss, training accuracy, validation loss, and validation accuracy in addition to the matrix containing these metrics at each epoch
    * validation loss and validation accuracy are a V x E matrix where E is number of epochs and V is number of sequences in the validation set, and contains the respective metric for each of the V sequences
  * **lengths.npy** - vector of lengths for the validation set for weighted sums, where sequences are in the same order as the validation loss and accuracy matrices
* **gap_predict_align.fa** - contains the sequences for the greedy prediction of the gap from both the left and right flanks (including the flank seeds), and the sequences from the input FASTA file 
* **my_model_weights.h5** - contains GapPredict model parameters and can be loaded into a GapPredict model

## Pipeline Reproduction Steps
Refer to this [link](https://github.com/bcgsc/GapPredict/tree/v1.0doc/scripts).

## Citations
1.	G. S. C. Slater and E. Birney. “Automated generation of heuristics for biological sequence comparison BMC Bioinform. Bioinform., vol. 6, no. 31, Feb. 2005.
2.	J. Chu, H. Mohamadi, E. Erhan, J. Tse, R. Chiu, S. Yeo, and I. Birol. “Improving on hash-based probabilistic sequence classification using multiple spaced seeds and multi-index Bloom filters”, bioRxiv:434795, Oct. 2018.
3.  D. Paulino, R. L. Warren, B. P. Vandervalk, A. Raymond, S. D. Jackman, and I. Birol. “Sealer: a scalable gap closing application for finishing draft genomes", BMC Bioinform., vol. 16, no. 230, Jul. 2015.
4.  E. Chen, J. Chu, J. Zhang, R. L. Warren, I. Birol. "GapPredict - A Language Model for Resolving Gaps in Draft Genome Assemblies", _IEEE/ACM Transactions on Computational Biology and Bioinformatics_. [doi:10.1109/TCBB.2021.3109557](http://dx.doi.org/10.1109/TCBB.2021.3109557)
