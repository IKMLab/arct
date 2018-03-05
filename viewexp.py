from arct import DBI
from ext import experiments
import argparse
import numpy as np
import pandas as pd
import os
import glovar


parser = argparse.ArgumentParser()
parser.add_argument('name',
                    type=str,
                    help='The name of the training run / experiment')
args = parser.parse_args()
name = getattr(args, 'name')
if not DBI.experiments.exists(name=name):
    info = 'Experiment with name %s not found. Existing names:\n' % name
    for n in [x['name'] for x in DBI.experiments.all()]:
        info += '\t%s\n' % n
    raise ValueError(info)
exp = DBI.experiments.get(name=name)
exp = experiments.adapt(exp)
print(exp)
print()
print('Means and Maxes:')
print('\tTrain')
print('\t\tMean: %s' % (np.average([y['train_acc'] for y in exp.results])))
print('\t\tMax: %s' % (np.max([y['train_acc'] for y in exp.results])))
print('\tTune')
print('\t\tMean: %s' % (np.average([y['tune_acc'] for y in exp.results])))
print('\t\tMax: %s' % (np.max([y['tune_acc'] for y in exp.results])))
df = pd.read_csv(os.path.join(glovar.DATA_DIR, 'test_evals.csv'))
print('\tTest')
print('\t\tMean: %s' % (df[df['file_name'].str.contains(name)]['result'].mean()))
print('\t\tMax: %s' % (df[df['file_name'].str.contains(name)]['result'].max()))
