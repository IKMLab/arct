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

## Table of Reported Experiment Results

| experiment_name   | model   | transfer   |   encoder_size |   train_acc |   tune_acc |   test_acc |
|:------------------|:--------|:-----------|---------------:|------------:|-----------:|-----------:|
| t2048fwcomp       | comp    | True       |           2048 |    0.724243 |   0.672552 |   0.560923 |
| r2048fwcomp       | comp    | False      |           2048 |    0.756793 |   0.668958 |   0.578153 |
| t1024fwcomp       | comp    | True       |           1024 |    0.780033 |   0.673906 |   0.574887 |
| r1024fwcomp       | comp    | False      |           1024 |    0.779005 |   0.673021 |   0.575788 |
| t512fwcomp        | comp    | True       |            512 |    0.874794 |   0.680104 |   0.613063 |
| t512fwcompN       | comp    | True       |            512 |    0.828421 |   0.680469 |   0.613063 |
| r512fwcomp        | comp    | False      |            512 |    0.797763 |   0.675729 |   0.569482 |
| t300fwcomp        | comp    | True       |            300 |    0.79324  |   0.669323 |   0.562275 |
| r300fwcomp        | comp    | False      |            300 |    0.819169 |   0.670521 |   0.576014 |
| t100fwcomp        | comp    | True       |            100 |    0.814655 |   0.672969 |   0.578829 |
| r100fwcomp        | comp    | False      |            100 |    0.848067 |   0.679479 |   0.588851 |
| t512fwcompc       | compc   | True       |            512 |    0.826373 |   0.67026  |   0.57545  |
| t512fwcompcHalf   | compc   | True       |            512 |    0.830099 |   0.63375  |   0.57545  |
| t512fwcompcN      | compc   | True       |            512 |    0.911398 |   0.654479 |   0.57545  |
| t640fwcomprw2     | comprw2 | True       |            640 |    0.819235 |   0.678385 |   0.580405 |
| t2048fwlin        | lin     | True       |           2048 |    0.756505 |   0.654063 |   0.538176 |
| r2048fwlin        | lin     | False      |           2048 |    0.726053 |   0.632708 |   0.527815 |
| t1024fwlin        | lin     | True       |           1024 |    0.684482 |   0.658854 |   0.523423 |
| r1024fwlin        | lin     | False      |           1024 |    0.708651 |   0.658958 |   0.538288 |
| t512fwlin         | lin     | True       |            512 |    0.668947 |   0.603594 |   0.509685 |
| r512fwlin         | lin     | False      |            512 |    0.721283 |   0.604844 |   0.526577 |
| t300fwlin         | lin     | True       |            300 |    0.639211 |   0.576406 |   0.504505 |
| r300fwlin         | lin     | False      |            300 |    0.69611  |   0.578542 |   0.519595 |
| t100fwlin         | lin     | True       |            100 |    0.57352  |   0.541458 |   0.514302 |
| r100fwlin         | lin     | False      |            100 |    0.575444 |   0.536094 |   0.512387 |


