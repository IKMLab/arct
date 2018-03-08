"""Script for all preprocessing and preparation.

Before running this script, be sure to set:
* glovar.ARCT_DIR: the folder in which the data files are kept
* glovar.GLOVE_PATH: path to 300d glove embeddings
"""
import arct
import os
import glovar
from ext import vocabulary, embeds, tokenization
from arct import factories


print('Preparing data...')


# 0. Make the data/ directory (it is excluded in .gitignore)
print('Creating data dir and subdirs...')
if not os.path.exists(glovar.DATA_DIR):
    os.makedirs(glovar.DATA_DIR)
if not os.path.exists(os.path.join(glovar.DATA_DIR, 'ckpts')):
    os.makedirs(os.path.join(glovar.DATA_DIR, 'ckpts'))
if not os.path.exists(os.path.join(glovar.DATA_DIR, 'logs')):
    os.makedirs(os.path.join(glovar.DATA_DIR, 'logs'))
if not os.path.exists(os.path.join(glovar.DATA_DIR, 'predictions')):
    os.makedirs(os.path.join(glovar.DATA_DIR, 'predictions'))


# 1. Create the Vocab
print('Creating vocab dict..')
if not arct.PKL.exists('vocab', ['arct']):
    words = factories.ARCTDataFactory().wordset(tokenization.SpaCyTokenizer())
    vocab = vocabulary.Vocab('arct', words)
    arct.PKL.save(vocab, 'vocab', ['arct'])
    print('Done.')
else:
    vocab = arct.PKL.load('vocab', ['arct'])
    print('Already exists.')


# 2. Create GloVe embeddings
print('Creating GloVe embeddings...')
if not arct.PKL.exists('glove', ['arct']):
    embeddings, oov = embeds.create_glove_embeddings(
        vocab.ix, 300, glovar.GLOVE_PATH)
    arct.PKL.save(embeddings, 'glove', ['arct'])
    arct.PKL.save(oov, 'glove_oov', ['arct'])
else:
    print('Already exists.')


print('Success.')
