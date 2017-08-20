"""For pre-processing the data."""
import pandas as pd
import os
from arct import glovar as gv
from ext import pickling as pkl
from ext import vocab_emb as ve


"""
4) need to pad the index sequences coming out of batching
5) then can try and re-run training
"""


FILES = {
    'dev': 'dev-full.txt',
    'train': 'train-full.txt',
    'swap': 'train-w-swap-full.txt'}
TEXT_COLS = ['warrant0', 'warrant1', 'reason', 'claim',
             'debateTitle', 'debateInfo']


def df(file_type):
    return pd.read_table(gv.DATA_DIR + FILES[file_type])


def df_text(d):
    return ' '.join([' '.join(d[col].values) for col in TEXT_COLS])


def embedding(emb_fam, emb_type, emb_size):
    emb_name = 'emb_%s_%s_%s.pkl' % (emb_fam, emb_type, emb_size)
    if os.path.exists(gv.DATA_DIR + emb_name):
        return pkl.load(gv.DATA_DIR, emb_name)
    emb_mat, _ = ve.create_embeddings(
        vocab(), emb_fam, emb_type, emb_size, gv.VEC_DIR[emb_fam])
    return emb_mat


def oov_counts():
    pkl_name = 'oov_counts.pkl'
    if os.path.exists(gv.DATA_DIR + pkl_name):
        return pkl.load(gv.DATA_DIR, pkl_name)
    counts = {}
    pkl.save(counts, gv.DATA_DIR, pkl_name)
    return counts


def text():
    dfs = [df('train'), df('dev')]
    texts = [df_text(d) for d in dfs]
    return ' '.join(texts)


def train_and_tune_data():
    df_train = df('train')
    df_tune = df('dev')
    cols = ['reason', 'claim', 'warrant0', 'warrant1', 'correctLabelW0orW1']
    train_data = df_train[cols].values
    tune_data = df_tune[cols].values
    return train_data, tune_data


def vocab():
    pkl_name = 'vocab_dict.pkl'
    if os.path.exists(gv.DATA_DIR + pkl_name):
        return pkl.load(gv.DATA_DIR, pkl_name)
    vocab_dict, _ = ve.create_vocab_dict(text())
    return vocab_dict
