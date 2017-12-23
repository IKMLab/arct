"""Global variables."""
import os


APP_DIR = os.getcwd()
CKPT_DIR = os.path.join(APP_DIR, 'ckpts/')
HOME_DIR = os.environ['HOME']
DATA_DIR = os.path.join(HOME_DIR, 'dev/data/')
GLOVE_DIR = os.path.join(HOME_DIR, 'dev/data/glove/glove.840B.300d.txt')
PKL_DIR = os.path.join(APP_DIR, 'pickles/')
