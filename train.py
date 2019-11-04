import csv
import numpy as np


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

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    print(sentences)

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print(x.shape, y.shape)

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()


