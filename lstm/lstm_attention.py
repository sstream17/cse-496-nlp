from pathlib import Path

import os
import random

import numpy as np
import matplotlib.pyplot  as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Dense, Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import json

from ArbitraryLearningRates import LearningRateCallback

class attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention,self).build(input_shape)
        
    def call(self, x):
        
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x,self.W)+self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return tf.keras.backend.sum(output, axis=1)

bag = []
if os.path.exists('all_lyrics.txt'):
    # Read all data from one file
    with open("all_lyrics.txt") as f:
        bag = f.readlines()
else:
    # Compile data from files
    for filename in os.listdir(r"lyrics"):
        with open("lyrics/" + filename, encoding="cp1252") as f:
            data = f.read().replace('–', '').replace('|','') + '}'#.encode('utf8')
            #print(data)
            
            # Parse data into structure
            # Returns a big dictionary
            dic = json.loads(data)
            
            # Combine the entries into a bag-of-lines
            for line in dic["Lyrics"][0]:
                # Don't want song structure or empty lines
                if line != '':
                    if line[0] != '[':
                        bag.append(line + ' ENDTOKEN')
                        #bag.append(line)
    with open("all_lyrics.txt", 'w') as f:
        for line in bag:
            f.write(line + '\r\n')
    
num_lines = len(bag)
print("{}: {}".format("Total number of lines", num_lines))

# Tokenize the bag
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' ')
tokenizer.fit_on_texts(bag)

# Mapping from words to integers in the vocabulary
word_to_integer = tokenizer.word_index
print(word_to_integer)
for t in word_to_integer:
    print("{0} -> {1}".format(t, word_to_integer[t]))

# Size of the vocabulary
num_words = len(word_to_integer.keys())+1
print("Total number of words: " +  str(num_words-1))

# Create reverse mapping from integers to words
integer_to_word = {val: key for key, val in word_to_integer.items()}
print(integer_to_word)

# Get the token-by-token data
input_sequences = []
for line in bag:
    # Get the integer sequence representation of the line
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(0, len(token_list)):
        # Take progressively larger slices of the input sequence (maybe remove?)
        n_gram_sequence = token_list[:i+1]
        
        # Add each slice to the input_sequences
        input_sequences.append(n_gram_sequence)
        
# We need to pad each sequence to be the maximum length
max_sequence_length = max([len(x) for x in input_sequences])-1
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length+1, padding = 'pre'))

# Print everything to get an idea of what it looks like
print(*(input_sequences[1:10,:].tolist()), sep='\n')
print(*(input_sequences[1:10,-1].tolist()))

# xs is the sequence except the last word
# ys is the last word in the sequence
# This gives us sequences matched to their next word, which hopefully the model can learn
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=num_words)
print(xs.shape)
print(ys.shape)

dim = max_sequence_length*2

# Layer descriptions
x_in = Input(shape=(max_sequence_length,), dtype='int32')
emb = tf.keras.layers.Embedding(num_words, dim, input_length=max_sequence_length)
x_emb = emb(x_in)
lstm1 = LSTM(dim, return_sequences=True)
x_seq_enc = lstm1(x_emb)
x_att = attention(return_sequences=True)(x_emb)
x_2 = Bidirectional(LSTM(dim, return_sequences=True))(x_att)
x_3 = Bidirectional(LSTM(dim, return_sequences=True))(x_2)
x_4 = Bidirectional(LSTM(dim))(x_3)
d = Dense(num_words, activation='softmax')(x_4)
out = Dropout(0.1)(d)

# Model time
model = Model(inputs=x_in, outputs=out)

# Model summary
model.summary()

# Compile model
opt = tf.keras.optimizers.Adam(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Start-stop a few times to try to work out of local minimums
learning_rates = [5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]
es = LearningRateCallback(rates=learning_rates, patience=15, restore_best_weights=False, early_stopping=False)

# Add a checkpoint callback
check = ModelCheckpoint('checkpoint', save_freq=10)
    
# Fit model, print only for epoch
history = model.fit(xs, ys, epochs=700, verbose=2, callbacks=[es])
    
# Save it in case we want to use this specific model later
model.save("model")

# Set up how our generated lyrics will look
next_words = max_sequence_length

# Could do them all together but it's easier to conceptualize this way
for j in range(0,20):
    seed = integer_to_word[random.randint(1,num_words)]
    for i in range(next_words):
        # Returns a list of lists, we just want one (the only)
        token_list = tokenizer.texts_to_sequences([seed])[0]
        
        # Pad the list
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding="pre")
        
        # Index of maximum in predictions
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        # find word corresponding to the maximal index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word == "endtoken":
            break
        seed = seed + " " + output_word

    print(seed)

fig, ax1 = plt.subplots()
ax2 = plt.twinx()

# Plot loss
ax1.plot(history.history['loss'], color='tab:blue')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')

# Plot accuracy
ax2.plot(history.history['accuracy'], color='tab:gray')
ax2.set_ylabel('Accuracy')

fig.legend()
fig.tight_layout()
fig.savefig('metrics.png')
























