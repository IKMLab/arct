"""For pre-processing the data."""
import pandas as pd
import os
import spacy
import collections
import pickle
import numpy as np


FILES = {
    'dev': 'dev-full.txt',
    'train': 'train-full.txt',
    'swap': 'train-w-swap-full.txt'}
GLOVE_DIR = '/home/hanshan/dev/data/glove/'
FAST_DIR = '/home/hanshan/dev/data/fasttext'
WORD_VECS = {
    'glove_twt': {
        25: GLOVE_DIR + 'glove.twitter.27B.25d.txt',
        50: GLOVE_DIR + 'glove.twitter.27B.50d.txt',
        100: GLOVE_DIR + 'glove.twitter.27B.100d.txt',
        200: GLOVE_DIR + 'glove.twitter.27B.200d.txt'},
    'glove_wpg': {
        50: GLOVE_DIR + 'glove.6B.50d.txt',
        100: GLOVE_DIR + 'glove.6B.100d.txt',
        200: GLOVE_DIR + 'glove.6B.200d.txt',
        300: GLOVE_DIR + 'glove.6B.300d.txt'},
    'glove_42B': {
        300: GLOVE_DIR + 'glove.42B.300d.txt'},
    'glove_840B': {
        300: GLOVE_DIR + 'glove.840B.300d.txt'},
    'fast_wiki': {
        300: FAST_DIR + ''
    },
    'fast_sub': {
        300: FAST_DIR + ''
    },
    'fast_crawl': {
        300: FAST_DIR + ''
    },
}
DATA_DIR = os.path.join(os.getcwd(), 'data/')
TEXT_COLS = ['warrant0', 'warrant1', 'reason', 'claim',
             'debateTitle', 'debateInfo']
NLP = spacy.load('en')
PADDING = "<PAD>"
UNKNOWN = "<UNK>"
LBR = '('
RBR = ')'


def create_embeddings(vocab, emb_type, emb_size):
    print('Creating word embeddings of type %s and dim %s...'
          % (emb_type, emb_size))
    vocab_size = max(vocab.values()) + 1
    counts = oov_counts()
    counts[emb_type] = vocab_size  # subtract as we find
    print('vocab_size = %s' % vocab_size)
    embeddings = np.random.normal(size=(vocab_size, emb_size))
    with open(WORD_VECS[emb_type][emb_size], 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab.keys():
                counts[emb_type] -= 1
                try:
                    embeddings[vocab[s[0]], :] = np.asarray(s[1:])
                except Exception as e:
                    print(vocab[s[0]])
                    print(len(vocab))
                    print(min(vocab.values()))
                    print(max(vocab.values()))
                    raise Exception('%s, %s:\n%s' % (i, s[0], repr(e)))
    save(embeddings, 'emb_%s_%s.pkl' % (emb_type, emb_size))
    save(counts, 'oov_counts.pkl')
    print('Success.')
    print('OOV count = %s' % counts[emb_type])
    return embeddings


def create_vocab():
    txt = text()
    doc = NLP(txt)
    counter = collections.Counter()
    counter.update([t.text for t in doc])
    save(counter, 'word_counter.pkl')
    tokens = set([t for t in counter] + [PADDING, UNKNOWN, LBR, RBR])
    vocab_dict = dict(zip(tokens, range(len(tokens))))
    save(vocab_dict, 'vocab_dict.pkl')
    return counter, vocab_dict


def df(file_type):
    # file_type in {dev, train, swap}
    return pd.read_table(DATA_DIR + FILES[file_type])


def df_text(d):
    return ' '.join([' '.join(d[col].values) for col in TEXT_COLS])


def load(pkl_name):
    file_path = pkl_path(pkl_name)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise Exception('Pickle not found: %s' % file_path)


def oov_counts():
    pkl_name = 'oov_counts.pkl'
    if os.path.exists(DATA_DIR + pkl_name):
        return load(pkl_name)
    counts = {}
    save(counts, pkl_name)
    return counts


def pkl_path(pkl_name):
    return os.path.join(DATA_DIR, pkl_name)


def save(obj, pkl_name):
    file_path = pkl_path(pkl_name)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def text():
    dfs = [df('train'), df('dev')]
    texts = [df_text(d) for d in dfs]
    return ' '.join(texts)


def vocab():
    return load('vocab_dict.pkl')
