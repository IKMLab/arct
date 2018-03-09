# NLITrans at SemEval 18 Task 12

This repository holds a frozen version of the code used for our
submission in SemEval 2018 Task 12: Argument Reasoning Comprehension
Test, as described in our paper (TODO: link here).

## Preparing for Usage

1. Required repdendencies are defined in `environment.yml`.
2. Set the `ARCT_DIR` and `GLOVE_DIR` variables in `glovar.py` to point
   to the folder with the ARCT data and GloVe embeddings file.
   See `glovar.py` for an example.
3. We used MongoDB to store experiment results and the like.
   We haven't unhooked this here, so you will need a local MongoDB
   instance running on localhost port 27017.
4. Once the environment is ready you will need to run `prepare.py`

## Reproducing Our Results

To reproduce any of our experiments, simply call the script
`reproduce.py {name}` where `name` corresponds to the `experiment_name`
column in the table below. This will perform training 20 times based on
the config settings we used (stored in `configs.py`, according to the
random seeds generated for our experiments. To try new random seeds add
the option `--new_seeds` - e.g. `reproduce.py t512fwcomp --new_seeds`.
It will save the results to `data/results.csv`. It will also print the
mean and max scores for each dataset at the end of the experiment.

Note: in this repository we supply the pre-trained encoders for 512,
and 640 encoder sizes. Those were the critical experiments. The 100 and
200 size encoders are also there, but the 1024 and 2048 are too big
for GitHub.

| experiment_name   | model   | transfer   |   encoder_size |   train_acc |   tune_acc |   test_acc |
|:------------------|:--------|:-----------|---------------:|------------:|-----------:|-----------:|
| t2048fwcomp       | Comp    | True       |           2048 |    0.724243 |   0.672552 |   0.560923 |
| r2048fwcomp       | Comp    | False      |           2048 |    0.756793 |   0.668958 |   0.578153 |
| t1024fwcomp       | Comp    | True       |           1024 |    0.780033 |   0.673906 |   0.574887 |
| r1024fwcomp       | Comp    | False      |           1024 |    0.779005 |   0.673021 |   0.575788 |
| t512fwcomp        | Comp    | True       |            512 |    0.874794 |   0.680104 |   0.613063 |
| t512fwcompN       | Comp    | True       |            512 |    0.828421 |   0.680469 |   0.613063 |
| r512fwcomp        | Comp    | False      |            512 |    0.797763 |   0.675729 |   0.569482 |
| t300fwcomp        | Comp    | True       |            300 |    0.79324  |   0.669323 |   0.562275 |
| r300fwcomp        | Comp    | False      |            300 |    0.819169 |   0.670521 |   0.576014 |
| t100fwcomp        | Comp    | True       |            100 |    0.814655 |   0.672969 |   0.578829 |
| r100fwcomp        | Comp    | False      |            100 |    0.848067 |   0.679479 |   0.588851 |
| t512fwcompc       | Comp-C  | True       |            512 |    0.826373 |   0.67026  |   0.57545  |
| t512fwcompcHalf   | Comp-C  | True       |            512 |    0.830099 |   0.63375  |   0.57545  |
| t512fwcompcN      | Comp-C  | True       |            512 |    0.911398 |   0.654479 |   0.57545  |
| t640fwcomprw2     | Comp-RW | True       |            640 |    0.819235 |   0.678385 |   0.580405 |
| t2048fwlin        | Lin     | True       |           2048 |    0.756505 |   0.654063 |   0.538176 |
| r2048fwlin        | Lin     | False      |           2048 |    0.726053 |   0.632708 |   0.527815 |
| t1024fwlin        | Lin     | True       |           1024 |    0.684482 |   0.658854 |   0.523423 |
| r1024fwlin        | Lin     | False      |           1024 |    0.708651 |   0.658958 |   0.538288 |
| t512fwlin         | Lin     | True       |            512 |    0.668947 |   0.603594 |   0.509685 |
| r512fwlin         | Lin     | False      |            512 |    0.721283 |   0.604844 |   0.526577 |
| t300fwlin         | Lin     | True       |            300 |    0.639211 |   0.576406 |   0.504505 |
| r300fwlin         | Lin     | False      |            300 |    0.69611  |   0.578542 |   0.519595 |
| t100fwlin         | Lin     | True       |            100 |    0.57352  |   0.541458 |   0.514302 |
| r100fwlin         | Lin     | False      |            100 |    0.575444 |   0.536094 |   0.512387 |


Additionally, a csv file with all run settings (including seed) and results
is located in `data/all_results.csv`.

