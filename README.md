# NLITrans at SemEval 18 Task 12

This repository holds a frozen version of the code used for our submission in SemEval 2018 Task 12: Argument Reasoning Comprehension Test, as described in our paper (TODO: link here).

## Preparing for Usage

1. Required repdendencies are defined in `environment.yml`.
2. Set the `ARCT_DIR` and `GLOVE_DIR` variables in `glovar.py` to point to the folder with the ARCT data and GloVe embeddings file. See the file for an example. 
3. Once the environment is ready you will need to run `prepare.py`

## Reproducing Our Results

[FUNTIONALITY UNDER CONSTRUCTION, SHOULD BE DONE TODAY]

The pre-trained encoders we used for sizes 512, 640, and 768 are provided,
for the critical experiments.

We have saved the best configuration settings for each model in the `config.py` file for reference.

All results in .csv format are provided in `/data/results.csv`.

To reproduce any of our experiments, simply call the script `reproduce.py --name` where `name` corresponds to the `experiment_name` column in the results csv file. This will perform training based on the config settings we used, and after training generate predictions on the training, dev, and testing sets. It will save these results to a .csv file in a global results csv file at `/data/experiment_results.csv`. It will also print the mean and max scores for each dataset at the end of the experiment.

