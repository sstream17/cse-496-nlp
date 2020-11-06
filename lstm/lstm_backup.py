from pathlib import Path

import os
import random

import numpy as np
import matplotlib.pyplot  as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as backend
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


import json


debug = False

bag = []
if debug:
    with open("remind.txt") as f:
        bag = f.readlines()
        #bag.append(f.readlines())
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


# Model time
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(num_words, max_sequence_length*2, input_length=max_sequence_length))
model.add(LSTM(int(np.floor(max_sequence_length*2)), return_sequences=True))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(Bidirectional(LSTM(int(np.floor(max_sequence_length*2)), return_sequences=True)))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(Bidirectional(LSTM(int(np.floor(max_sequence_length*2)), return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Bidirectional(LSTM(int(np.floor(max_sequence_length*2)))))
model.add(tf.keras.layers.Dropout(0.2))
#model.add(Dense(num_words/10, activation="softmax"))
#model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(num_words, activation="softmax"))
model.add(tf.keras.layers.Dropout(0.2))

# Model summary
model.summary()

# Compile model
learning_rates = [0.005, 0.001, 0.0005]
def LearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, rates=[0.005, 0.001, 0.0005], patience=25, use_best_rates = False):
        super(LearningRateCallback, self).__init__()
        self.rates = rates
        self.rate_index = 0
        self.patience = patience
        self.use_best = use_best_rates # flag
        self.best_weights = None
    
    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            if self.use_best:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.rate_index < len(self.rates) - 1:
                    self.rate_index += 1
                    backend.set_value(model.optimizer.lr, learning_rates[p])
                    print("Changing rate to " + str(self.rates[self.rate_index]) + " on epoch " + str(epoch))
                else:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.use_best:
                        self.model.set_weights(self.best_weights)
                        print("Early stopping ")
    
    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Ending early on epoch" + str(stopped_epoch))

opt = tf.keras.optimizers.Adam(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Start-stop a few times to try to work out of local minimums
init_epoch = 0
passes = 3
history = []
for p in range(0,passes):
    print("Pass " + str(p) + "; learning rate: " + str(learning_rates[p]))
    # Establish early-stopping based on loss
    es = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=25)
    
    # Select learning rate for pass
    backend.set_value(model.optimizer.lr, learning_rates[p])
    
    # Fit model, print only for epoch
    history.append(model.fit(xs, ys, epochs=init_epoch+200, initial_epoch=init_epoch, verbose=2, callbacks=[es]))
    
    # Save it in case we want to use this specific model later
    model.save("model")

    # Set up how our generated lyrics will look
    seed_text = [integer_to_word[random.randint(1,num_words)], integer_to_word[random.randint(1,num_words)], integer_to_word[random.randint(1,num_words)], integer_to_word[random.randint(1,num_words)]]
    next_words = max_sequence_length

    # Could do them all together but it's easier to conceptualize this way
    for part in seed_text:
        seed = part
        for i in range(next_words):
            # Returns a list of lists, we just want one (the only)
            token_list = tokenizer.texts_to_sequences([seed])[0]
            
            # Pad the list
            token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding="pre")
            
            # Index of maximum in predictions
            predicted = np.argmax(model.predict(token_list), axis=-1)
            
            # Debug print
            #x = model.predict(token_list)[0]
            #if i == next_words-1:
            #    for tok in x:
            #        print(tok, end=" ")
            #    print('')
            
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

big_hist = history[0]
for p in range(1,passes):
    big_hist.history['loss'] += history[p].history['loss']
    big_hist.history['accuracy']+= history[p].history['accuracy']

plt.plot(big_hist.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss.png')

plt.plot(big_hist.history['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('accuracy.png')
























