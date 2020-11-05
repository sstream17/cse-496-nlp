from pathlib import Path

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import json

import arch_utils
import utils
from layers import *


debug = False

bag = []
if debug:
    with open("remind.txt") as f:
        bag = f.readlines()
        # bag.append(f.readlines())
    with open("photograph.txt") as f:
        bag = bag + f.readlines()
elif os.path.exists('all_lyrics.txt'):
    # Read all data from one file
    with open("all_lyrics.txt") as f:
        bag = f.readlines()
else:
    # Compile data from files
    for filename in os.listdir(r"lyrics"):
        with open("lyrics/" + filename, encoding="cp1252") as f:
            data = f.read().replace('–', '').replace('|', '') + '}'  # .encode('utf8')
            print(data)

            # Parse data into structure
            # Returns a big dictionary
            dic = json.loads(data)

            # Combine the entries into a bag-of-lines
            for line in dic["Lyrics"][0]:
                # Don't want song structure or empty lines
                if line != '':
                    if line[0] != '[':
                        #bag.append('STARTTOKEN ' + line + ' ENDTOKEN')
                        bag.append(line)
    with open("all_lyrics.txt", 'w') as f:
        for line in bag:
            f.write(line + '\r\n')

num_lines = len(bag)
print("{}: {}".format("Total number of lines", num_lines))

#b = ""
# for i in range(0,80):
#    line = bag[i]
#    print(line)
#    b = b + line

# Tokenize the bag
tokenizer = Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ')
tokenizer.fit_on_texts(bag)

# Mapping between words and integers in the vocabulary
word_index = tokenizer.word_index
print(word_index)
for t in word_index:
    print("{0} -> {1}".format(t, word_index[t]))
# Size of the vocabulary
num_words = len(word_index.keys())+1
print("{:50s}: {}".format("Total number of words", num_words-1))

input_sequences = []
for line in bag:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_length, padding='post'))

print(input_sequences)
print(input_sequences[:, 0])
xs, labels = input_sequences[:, :], input_sequences[:, 0]
ys = tf.keras.utils.to_categorical(labels, num_classes=num_words)
print(xs.shape)
print(ys.shape)


# LSTM time (remake once we have a handle on it)
right = LSTM(int(np.floor(max_sequence_length)), return_sequences=True)
left = Bidirectional(LSTM(int(np.floor(max_sequence_length/4))))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(
    num_words, 32, input_length=max_sequence_length))
model.add(LSTM(int(np.floor(max_sequence_length/2)), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.05))
model.add(Bidirectional(
    LSTM(int(np.floor(max_sequence_length/2)), return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.05))
model.add(Bidirectional(
    LSTM(int(np.floor(max_sequence_length/4)), return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.05))
model.add(Bidirectional(LSTM(int(np.floor(max_sequence_length/4)))))
model.add(tf.keras.layers.Dropout(0.05))
#model.add(Dense(num_words/10, activation="softmax"))
# model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(num_words, activation="softmax"))
model.add(tf.keras.layers.Dropout(0.1))


model.summary()

# Change later as well
opt = tf.keras.optimizers.Adam(lr=0.2)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=15)
model.fit(xs, ys, epochs=150, callbacks=[es])

model.save("model")

seed_text = ["run", "photograph", "turn", "wish"]
next_words = 20

# could do them all together but it's easier to conceptualize this way
for part in seed_text:
    seed = part
    for i in range(next_words):
        # returns a list of lists, we just want one (the only)
        token_list = tokenizer.texts_to_sequences([seed])[0]

        # pad the list
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_length, padding="post")

        # index of maximum in predictions
        predicted = np.argmax(model.predict(token_list), axis=-1)

        # debug print
        if i == next_words-1:
            print(model.predict(token_list))

        # find word corresponding to the maximal index
        # better way to do this: reverse the dictionary relating indices to words
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed = seed + " " + output_word

    print(seed.replace("STARTTOKEN ", ""))
