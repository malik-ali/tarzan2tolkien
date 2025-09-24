import sys
import os

raise NotImplementedError("Set ROOT_DIR to be the absolute path to root of your repo")
## --- UPDATE --- ##
ROOT_DIR = '../'
# sys.path.append(ROOT_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR = os.path.join(ROOT_DIR, 'out')
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')

## --- UPDATE --- ##
# This file points to a list of cefr words with word frequency ranks.
# We weren't able to share this file but it should be easy to find a similar one online.
# You will have to adjust the code in src/featurise.py:load_word_data appropriately.
WORD_FILE = os.path.join(DATA_DIR, 'en', 'cefr_words.csv')

## --- UPDATE --- ##
# Download this from huggingface
TINYTOLKIEN_PATH = os.path.join(DATA_DIR, 'opensource', 'tinytolkien.csv')

