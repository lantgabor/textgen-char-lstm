import csv
import sys

import numpy as np
import re
import random

# Open the train data csv file
from matplotlib import pyplot

with open('./data/hackernews.csv', 'r', encoding='UTF8') as fp:
    reader = csv.reader(fp)

    # Remove unwanted symbols from the text
    text = ''
    corpus = []
    for row in reader:
        row[0] = re.sub('[^A-Za-z0-9 ]+', '', row[0])
        text += row[0]
        corpus.append(row[0])

    # Analyze the data
    print('total length of all the text:', len(text))
    print('number of sentences:', len(corpus))

    chars = sorted(list(set(text)))
    print('number of different characters:', len(chars))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('chars -> num:', indices_char)

    # Move the text into windowed format. Use the maxlen as the window size for each sentence.
    # The window step is set by the step variable
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Number of training data:', len(sentences))
    rand_ind = 1234
    print('One training example:', sentences[rand_ind], next_chars[rand_ind])

    # Create tensors from training data
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    # One-hot encode training data
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

print ('Input shape:', x.shape)
print ('Output shape:', y.shape)
print (x[rand_ind])
print (y[rand_ind])

# Create the Model

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

patience=5
early_stopping=EarlyStopping(patience=patience, verbose=1)

checkpointer=ModelCheckpoint(filepath='weights.hdf5', save_best_only=True, verbose=1)

# LSTM model generates text charcacter by character
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Helper function to correct the prediciton each time
# @preds: output neurons
# @temperature: 1.0 is the most conservative, 0.0 is the most confident (willing to make spelling and other errors).
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    print('Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for temperature in [0.2, 0.5, 1.0]:
        print('temperature:', temperature)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(40):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# Fit the model
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

history = model.fit(x, y,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback, checkpointer, early_stopping],verbose=1, validation_split=0.2)

# plot train and validation loss
pyplot.plot(history.history['loss'][500:])
pyplot.plot(history.history['val_loss'][500:])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()