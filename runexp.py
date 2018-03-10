"""Script for running a single experiment."""
import argparse
from ext import argparsing, processing
from arct import configuration, GRID_MANAGER as gm
import arct


parser = argparse.ArgumentParser()
parser.add_argument('name',
                    type=str,
                    help='The name of the training run / experiment')
parser.add_argument('seed',
                    type=int,
                    help='The random seed to use')
args = parser.parse_args()
name = getattr(args, 'name')
seed = getattr(args, 'seed')


config = argparsing.parse(configuration.Config())
if config['from_grid']:
    seed = config['seed']
    n_runs = config['n_runs']
    transfer = config['from_grid_trans']
    name = config['name']
    train_ss = config['train_subsample']
    target = config['target']
    _, config = gm.best(config['grid_name'])
    config['seed'] = seed
    config['n_runs'] = n_runs
    config['transfer'] = transfer
    config['from_grid_trans'] = transfer
    config['name'] = name
    config['train_subsample'] = train_ss
    config['target'] = target
config = configuration.Config(base_config=config)
processor = processing.Processor(arct.TRAIN_FACTORY)
processor.run_exp(config)