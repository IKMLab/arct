"""Script for reproducing experiments."""
import argparse
from ext import processing
import arct
import configs
from arct import configuration
import pandas as pd
import glovar
import os


parser = argparse.ArgumentParser()
parser.add_argument('name',
                    type=str,
                    help='The name of the training run / experiment')
parser.add_argument('--new_seeds',
                    action='store_true',
                    help='The name of the training run / experiment')
args = parser.parse_args()
name = getattr(args, 'name')
new_seeds = getattr(args, 'new_seeds')


config = configuration.Config(configs.get_config(name))
config['n_runs'] = 20
processor = processing.Processor(arct.TRAIN_FACTORY, new_seeds)
experiment = processor.run_exp(config)


print('Appending to results.csv...')
file_path = os.path.join(glovar.DATA_DIR, 'results.csv')
new_data = {
    'model': [],
    'encoder_size': [],
    'transfer': [],
    'train_acc': [],
    'tune_acc': [],
    'test_acc': []}
for r in experiment.results:
    new_data['model'].append(experiment.model)
    new_data['encoder_size'].append(experiment.config['encoder_size'])
    new_data['transfer'].append(experiment.config['transfer'])
    new_data['train_acc'].append(r['train_acc'])
    new_data['tune_acc'].append(r['tune_acc'])
    new_data['test_acc'].append(r['test_acc'])
df_new = pd.DataFrame(new_data)
if os.path.exists(file_path):
    df_old = pd.read_csv(file_path)
    df_new = df_old.append(df_new)
df_new.to_csv(file_path, index=False)
print('Success.')
