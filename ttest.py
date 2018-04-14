"""Quick script for t-test."""
from scipy import stats
import pandas as pd
import os
import glovar
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('name1',
                    type=str,
                    help='The name of the first training run / experiment')
parser.add_argument('name2',
                    type=str,
                    help='The name of the second training run / experiment')
parser.add_argument('--take',
                    type=int,
                    default=200,
                    help='How many of the records to take')
args = parser.parse_args()
name1 = getattr(args, 'name1')
name2 = getattr(args, 'name2')
take = getattr(args, 'take')


file_path = os.path.join(glovar.DATA_DIR, 'results.csv')
df = pd.read_csv(file_path)
df1 = df[df['experiment_name'] == name1]
df2 = df[df['experiment_name'] == name2]
print('Number of records (1): %s' % len(df1))
print('Number of records (2): %s' % len(df2))
print('Taking %s of those...' % take)
df1 = df1.iloc[0:take]
df2 = df2.iloc[0:take]
v1 = df1['test_acc'].values
v2 = df2['test_acc'].values
print(stats.ttest_ind(v1, v2, equal_var=False))
