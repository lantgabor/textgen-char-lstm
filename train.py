import csv

with open('C:\\Users\\lantg\\PycharmProjects\\textgen-char-lstm\\data\\politics.csv', 'r', encoding='UTF8') as fp:
    reader = csv.reader(fp)

    text = ''
    corpus = []

    for row in reader:
        row[0] = row[0].replace(r'[^\x00-\x7F]', r'')
        text += row[0]
        corpus.append(row[0])

    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    print(chars)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import re




