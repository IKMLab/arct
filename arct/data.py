"""Argument Reasoning Comprehension Test data."""
import pandas as pd
import glovar
from ext import PKL, torch_util, vocab, embeds
import numpy as np
import nltk
import os


FILES = {
    'dev': 'dev-full.txt',
    'dev_eval': 'dev-only-data.txt',
    'train': 'train-full.txt',
    'swap': 'train-w-swap-full.txt'}
TEXT_COLS = ['warrant0', 'warrant1', 'reason', 'claim',
             'debateTitle', 'debateInfo']


def df(file_type):
    return pd.read_table(glovar.DATA_DIR + 'argmin/arct/' + FILES[file_type])


def df_text(d):
    return ' '.join([' '.join(d[col].values) for col in TEXT_COLS])


def create_vocab_dict():
    tokens = get_tokens()
    vocab_dict, counter = vocab.create_vocab_dict(tokens)
    PKL.save(vocab_dict, 'arct_vocab_dict.pkl')
    PKL.save(counter, 'arct_counter.pkl')
    return vocab_dict, counter


def create_glove_embeddings():
    vocab_dict = get_vocab_dict()
    embeddings, oov = embeds.create_embeddings(
        vocab_dict, 300, glovar.GLOVE_DIR)
    PKL.save(embeddings, 'arct_glove_embeddings.pkl')
    PKL.save(oov, 'arct_oov.pkl')
    return vocab_dict, oov


def get_vocab_dict():
    return PKL.load('arct_vocab_dict.pkl')


def get_glove_embeddings():
    return PKL.load('arct_glove_embeddings.pkl')


def get_text():
    dfs = [df('train'), df('dev'), df('dev_eval')]
    texts = [df_text(d) for d in dfs]
    return ' '.join(texts)


def get_tokens():
    text = get_text()
    return nltk.word_tokenize(text)


def get_sentences():
    sents = ''
    all_data = adapt(df('train')) + adapt(df('dev')) + adapt(df('dev_eval'), False)
    for x in all_data:
        for col in TEXT_COLS:
            sents += x[col] + '\n'
    file_path = os.path.join(glovar.APP_DIR, 'data/arct_sents.txt')
    with open(file_path, 'w+') as f:
        f.write(sents)
    return sents


def load_data():
    df_train = df('train')
    df_tune = df('dev')
    train = adapt(df_train)
    tune = adapt(df_tune)
    return train, tune


def adapt(df, has_label=True):
    if has_label:
        data = [{'claim': x[1]['claim'],
                 'reason': x[1]['reason'],
                 'warrant0': x[1]['warrant0'],
                 'warrant1': x[1]['warrant1'],
                 'label': x[1]['correctLabelW0orW1'],
                 'id': x[1]['#id'],
                 'debateTitle': x[1]['debateTitle'],
                 'debateInfo': x[1]['debateInfo']}
                for x in df.iterrows()]
    else:
        data = [{'claim': x[1]['claim'],
                 'reason': x[1]['reason'],
                 'warrant0': x[1]['warrant0'],
                 'warrant1': x[1]['warrant1'],
                 'id': x[1]['#id'],
                 'debateTitle': x[1]['debateTitle'],
                 'debateInfo': x[1]['debateInfo']}
               for x in df.iterrows()]
    return data


def eval_data():
    df_dev_eval = df('dev_eval')
    dev_eval = adapt(df_dev_eval, False)
    return dev_eval


class RNNBatch:
    """Wrapper for a batch of ARCT data."""

    def __init__(self, claims, claim_lens, claims_rev_ix_sort,
                 reasons, reason_lens, reasons_rev_ix_sort,
                 warrant0s, warrant0_lens, warrant0s_rev_ix_sort,
                 warrant1s, warrant1_lens, warrant1s_rev_ix_sort,
                 labels, ids):
        self.claims = claims
        self.claim_lens = claim_lens
        self.claims_rev_ix_sort = claims_rev_ix_sort
        self.reasons = reasons
        self.reason_lens = reason_lens
        self.reasons_rev_ix_sort = reasons_rev_ix_sort
        self.warrant0s = warrant0s
        self.warrant0_lens = warrant0_lens
        self.warrant0s_rev_ix_sort = warrant0s_rev_ix_sort
        self.warrant1s = warrant1s
        self.warrant1_lens = warrant1_lens
        self.warrant1s_rev_ix_sort = warrant1s_rev_ix_sort
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.claims)


class CollateForRNN:
    """Wrapping collate functions for NLI data for PyTorch RNN encoders."""

    def __init__(self, vocab_dict):
        self.vocab_dict = vocab_dict
        self.prep4rnn = torch_util.CollateForRNN(vocab_dict)

    def __call__(self, data):
        # data is a batch of original JSON NLI data
        claims = [nltk.word_tokenize(x['claim']) for x in data]
        reasons = [nltk.word_tokenize(x['reason']) for x in data]
        warrant0s = [nltk.word_tokenize(x['warrant0']) for x in data]
        warrant1s = [nltk.word_tokenize(x['warrant1']) for x in data]
        if 'label' in data[0].keys():
            labels = np.array([x['label'] for x in data])
        else:
            labels = None
        ids = [x['id'] for x in data]
        claims, claim_lens, claims_rev_ix_sort = self.prep4rnn(claims)
        reasons, reason_lens, reasons_rev_ix_sort = self.prep4rnn(reasons)
        warrant0s, warrant0_lens, warrant0s_rev_ix_sort = self.prep4rnn(
            warrant0s)
        warrant1s, warrant1_lens, warrant1s_rev_ix_sort = self.prep4rnn(
            warrant1s)
        return RNNBatch(claims, claim_lens, claims_rev_ix_sort,
                        reasons, reason_lens, reasons_rev_ix_sort,
                        warrant0s, warrant0_lens, warrant0s_rev_ix_sort,
                        warrant1s, warrant1_lens, warrant1s_rev_ix_sort,
                        labels, ids)
