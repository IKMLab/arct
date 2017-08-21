"""Global Variables."""
import os


ROOT_DIR = 'd:/'
#ROOT_DIR = '/home/hanshan/'
VEC_DIR = {
    'glove': ROOT_DIR + 'dev/data/glove/',
    'fast': ROOT_DIR + 'dev/data/fasttext/'}
DATA_DIR = os.path.join(os.getcwd(), 'data/')
MODELS = ['bilstm1', 'bilstm2']
CKPT_DIR = os.path.join(DATA_DIR, 'ckpts/')
