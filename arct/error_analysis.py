"""Tools for error analysis between models.

* For each model (experiment name) we have 20 prediction files
* For each prediction file, generate sets of correct and incorrect ids
* From those, determine the set of always correct ids
* Compare the always correct ids of two models to examine their capabilities
"""
from arct import EXPERIMENT_MANAGER as em, DATA_FACTORY as df
import glovar
import os


def run_sets(name):
    """Get correct and incorrect id sets for all runs in an experiment.

    Args:
      name: String.

    Returns:
      Dictionary. Keys are the names of prediction files. Values are another
        dictionary with correct and incorrect as keys, and the corresponding
        sets as values.
    """
    sets = {}
    test_labels = df.test_labels()
    dir_path = os.path.join(glovar.DATA_DIR, 'predictions')
    file_names = [n for n in os.listdir(dir_path) if name in n]
    for file_name in file_names:
        sets[file_name] = {'correct': set([]), 'incorrect': set([])}
        with open(os.path.join(dir_path, file_name)) as f:
            for line in f.readlines():
                id, pred = line.split('\t')
                if int(pred.strip()) == test_labels[id]:
                    sets[file_name]['correct'].update([id])
                else:
                    sets[file_name]['incorrect'].update([id])
    return sets


def correct_ids(run_sets):
    """Get the set of ids that are always correct over runs.

    Args:
      run_sets: Dictionary. The return value of run_sets().

    Returns:
      Set.
    """
    test_labels = df.test_labels()
    corrects = set(test_labels.keys())
    for sets in run_sets.values():
        corrects = corrects.intersection(sets['correct'])
    return corrects


def incorrect_ids(run_sets):
    """Get the set of ids that are always incorrect over runs.

    Args:
      run_sets: Dictionary. The return value of run_sets().

    Returns:
      Set.
    """
    test_labels = df.test_labels()
    corrects = set(test_labels.keys())
    for sets in run_sets.values():
        corrects = corrects.intersection(sets['incorrect'])
    return corrects


def compare(name_better, name_worse):
    """Compare two runs.

    Args:
      name_better: String, the name of the experiment with better results.
      name_worse: String, the name of the experiment with worse results.

    Returns:
      ?
    """
    test_data = df.test(None)
    test_labels = df.test_labels()
    n = len(test_data)
    better_sets = run_sets(name_better)
    worse_sets = run_sets(name_worse)
    better_corrects = correct_ids(better_sets)
    worse_corrects = correct_ids(worse_sets)
    better_incorrects = incorrect_ids(better_sets)
    worse_incorrects = incorrect_ids(worse_sets)
    print('No. always correct %s: %s' % (name_better, len(better_corrects)))
    print('No. always correct %s: %s' % (name_worse, len(worse_corrects)))
    print('No. always incorrect %s: %s' % (name_better, len(better_incorrects)))
    print('No. always incorrect %s: %s' % (name_worse, len(worse_incorrects)))
    print('%% always correct %s: %s' % (name_better, len(better_corrects) / n))
    print('%% always correct %s: %s' % (name_worse, len(worse_corrects) / n))
    print('%% always incorrect %s: %s' % (name_better, len(better_incorrects) / n))
    print('%% always incorrect %s: %s' % (name_worse, len(worse_incorrects) / n))
    print('The following samples are the difference in always corrects:')
    diff = better_corrects - worse_corrects
    for id in diff:
        sample = next(x for x in test_data if x['id'] == id)
        print_sample(sample, test_labels[id])


def print_sample(x, label):
    print('----------')
    print('Reason:\t%s' % x['reason'])
    print('Claim: \t%s' % x['claim'])
    print('W0:    \t%s' % x['warrant0'])
    print('W1:    \t%s' % x['warrant1'])
    print('Label: \t%s' % label)
