"""Script for running a single experiment."""
import argparse
from ext import processing
from arct import configuration
import arct
import configs


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


config = configuration.Config(configs.get_config(name))
config['n_runs'] = 20
config['seed'] = seed
config['name'] = name + '_seed_%s' % seed
config['n_runs'] = 1
processor = processing.Processor(arct.TRAIN_FACTORY, False)
experiment = processor.run_exp(config)
