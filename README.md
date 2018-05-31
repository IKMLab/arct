# NLITrans at SemEval 18 Task 12

This repository holds a frozen version of the code used for our
submission in SemEval 2018 Task 12: Argument Reasoning Comprehension
Test, as described in our paper (https://arxiv.org/abs/1804.08266).

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
It will save the results to `data/results.csv`. It will also display the
mean and max scores for each dataset at the end of the experiment.
If you want a different number of runs, add the argument `--n_runs` to
the command.

Note: in this repository we supply the pre-trained encoders for 512,
and 640 encoder sizes. Those were the critical experiments. The 100 and
200 size encoders are also there, but the 1024 and 2048 are too big
for GitHub. "compX" is our submission model.

The table below gives mean accuracies over 200 runs.

| experiment_name   | model   | transfer   |   encoder_size |   train_acc |   tune_acc |   test_acc |
|:------------------|:--------|:-----------|---------------:|------------:|-----------:|-----------:|
| compX             | comp    | True       |           2048 |    0.755493 |   0.6725   |   0.592448 |
| t2048fwcomp       | comp    | True       |           2048 |    0.730263 |   0.670969 |   0.601823 |
| r2048fwcomp       | comp    | False      |           2048 |    0.732673 |   0.672057 |   0.598586 |
| t1024fwcomp       | comp    | True       |           1024 |    0.783377 |   0.674703 |   0.602470 |
| r1024fwcomp       | comp    | False      |           1024 |    0.786148 |   0.673453 |   0.605848 |
| t512fwcomp        | comp    | True       |            512 |    0.880679 |   0.679781 |   0.644263 |
| t512fwcompHalf    | comp    | True       |            512 |    0.892463 |   0.668260 |   0.633181 |
| t512fwcompN       | comp    | True       |            512 |    0.910945 |   0.676214 |   0.635260 |
| r512fwcomp        | comp    | False      |            512 |    0.797564 |   0.671818 |   0.618110 |
| t300fwcomp        | comp    | True       |            300 |    0.811187 |   0.671240 |   0.626012 |
| r300fwcomp        | comp    | False      |            300 |    0.826630 |   0.674359 |   0.628482 |
| t100fwcomp        | comp    | True       |            100 |    0.819822 |   0.670724 |   0.632883 |
| r100fwcomp        | comp    | False      |            100 |    0.837838 |   0.674276 |   0.631019 |
| t512fwcompc       | compc   | True       |            512 |    0.815479 |   0.666417 |   0.591231 |
| t512fwcompcHalf   | compc   | True       |            512 |    0.815479 |   0.666417 |   0.591231 |
| t512fwcompcN      | compc   | True       |            512 |    0.936771 |   0.650984 |   0.574981 |
| t640fwcomprw2     | comprw2 | True       |            640 |    0.808567 |   0.676740 |   0.605982 |

Additionally, a csv file with all run settings (including seed) and results
is located in `data/all_results.csv`.

