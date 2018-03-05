"""Global Variables."""
import os


APP_DIR = os.getcwd()
DATA_DIR = os.path.join(APP_DIR, 'data')
HOME_DIR = os.environ['HOME']
ARCT_DIR = os.path.join(HOME_DIR, 'dev/data/argmin/arct')
GLOVE_PATH = os.path.join(HOME_DIR, 'dev/data/glove/glove.840B.300d.txt')
FASTTEXT_PATH = os.path.join(HOME_DIR, 'dev/data/fasttext/crawl-300d-2M.vec')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
CKPT_DIR = os.path.join(DATA_DIR, 'ckpts')
