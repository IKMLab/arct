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
                    help='Whether or not to use new seeds')
parser.add_argument('--n_runs',
                    type=int,
                    default=20,
                    help='The name of the training run / experiment')
args = parser.parse_args()
name = getattr(args, 'name')
new_seeds = getattr(args, 'new_seeds')
n_runs = getattr(args, 'n_runs')


config = configuration.Config(configs.get_config(name))
config['n_runs'] = n_runs
processor = processing.Processor(arct.TRAIN_FACTORY, new_seeds)
experiment = processor.run_exp(config)
print(experiment)


print('Appending to results.csv...')
file_path = os.path.join(glovar.DATA_DIR, 'results.csv')
new_data = {
    'experiment_name': [],
    'model': [],
    'encoder_size': [],
    'transfer': [],
    'seed': [],
    'train_acc': [],
    'tune_acc': [],
    'test_acc': []}
for r in experiment.results:
    new_data['experiment_name'].append(name)
    new_data['model'].append(experiment.model)
    new_data['encoder_size'].append(experiment.config['encoder_size'])
    new_data['transfer'].append(experiment.config['transfer'])
    new_data['seed'].append(r['seed'])
    new_data['train_acc'].append(r['train_acc'])
    new_data['tune_acc'].append(r['tune_acc'])
    new_data['test_acc'].append(r['test_acc'])
df_new = pd.DataFrame(new_data)
if os.path.exists(file_path):
    df_old = pd.read_csv(file_path)
    df_new = df_old.append(df_new)
df_new.to_csv(file_path, index=False)
print('Success.')
