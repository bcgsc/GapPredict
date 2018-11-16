# Sealer_NN
LSTM approach to predicting the next base(s) for a given kmer.

Implemented in Python 3.6

To use, run:
`python src/app/<file name> <FASTA1> <FASTA2> ...`

`FASTAn` = path to a FASTA file to use in the training set (optional, we use whatever is hardcoded in the repository if no files are given)

`<file name>` is one of:
`app_dummy.py` = randomly predicts each base, mainly used to benchmark the preprocessing steps
`app_keras.py` = trains a model with k+d-mers from the input reads, then calculates training and validation error
`app_keras_load_Weights.py` = loads model weights from the default location (`src/app/weights/my_model_Weights.h5`) and calculates training and validation error 

You can also run:
`python src/app/interactive_predictor.py`

To open a command line application that predicts a fixed number of bases for a fixed kmer length based on the most recent trained model at the default location.

Parameterizing these apps are still a task to be done.
