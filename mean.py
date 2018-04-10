"""Script for quick checking of a mean."""
import pandas as pd
import glovar
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('name',
                    type=str,
                    help='The name of the training run / experiment')
parser.add_argument('--take',
                    type=int,
                    default=200,
                    help='How many of the records to take')
args = parser.parse_args()
name = getattr(args, 'name')
take = getattr(args, 'take')


file_path = os.path.join(glovar.DATA_DIR, 'results.csv')
df = pd.read_csv(file_path)
df = df[df['experiment_name'] == name]
print('Number of records: %s' % len(df))
print('Taking %s of those...' % take)
print(df.iloc[0:take].mean())
