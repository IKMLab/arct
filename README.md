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
for GitHub. "compX" is our submission model.

| experiment_name   | model   | transfer   |   encoder_size |   train_acc |   tune_acc |   test_acc |
|:------------------|:--------|:-----------|---------------:|------------:|-----------:|-----------:|
| compX             | comp    | True       |           2048 |    0.755493 |   0.6725   |   0.592448 |
| t2048fwcomp       | comp    | True       |           2048 |    0.721834 |   0.669635 |   0.601823 |
| r2048fwcomp       | comp    | False      |           2048 |    0.731719 |   0.676302 |   0.604278 |
| t1024fwcomp       | comp    | True       |           1024 |    0.808289 |   0.672969 |   0.597396 |
| r1024fwcomp       | comp    | False      |           1024 |    0.79588  |   0.674688 |   0.601525 |
| t512fwcomp        | comp    | True       |            512 |    0.906743 |   0.68125  |   0.645833 |
| t512fwcompHalf    | comp    | True       |            512 |    0.856324 |   0.681406 |   0.646205 |
| t512fwcompN       | comp    | True       |            512 |    0.854252 |   0.678646 |   0.646949 |
| r512fwcomp        | comp    | False      |            512 |    0.806349 |   0.671771 |   0.62247  |
| t300fwcomp        | comp    | True       |            300 |    0.781242 |   0.673177 |   0.62872  |
| r300fwcomp        | comp    | False      |            300 |    0.775354 |   0.670885 |   0.635231 |
| t100fwcomp        | comp    | True       |            100 |    0.84574  |   0.67651  |   0.640513 |
| r100fwcomp        | comp    | False      |            100 |    0.833289 |   0.673073 |   0.632254 |
| t512fwcompc       | compc   | True       |            512 |    0.839753 |   0.668854 |   0.588988 |
| t512fwcompcHalf   | compc   | True       |            512 |    0.824128 |   0.668021 |   0.586235 |
| t512fwcompcN      | compc   | True       |            512 |    0.909531 |   0.651302 |   0.570164 |
| t640fwcomprw2     | comprw2 | True       |            640 |    0.824137 |   0.675156 |   0.609263 |

Additionally, a csv file with all run settings (including seed) and results
is located in `data/all_results.csv`.

