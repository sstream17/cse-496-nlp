import json
import os
import random
import sys
import numpy as np


def remove_punctuation(word):
    return word \
        .replace(',', '') \
        .replace("'", '') \
        .replace('(', '') \
        .replace(')', '') \
        .lower()


def even_probabilities(probabilities, _):
    probabilities = probabilities + [0]
    return [1 / len(probabilities) for i in probabilities]


def weighted_probabilities(current_probabilities, number_keys):
    return current_probabilities + [1 / (number_keys * 10)]

        
def add_pair(lyrics, key, word, calculate_probabilities):
    key = remove_punctuation(key)
    word = remove_punctuation(word)
    
    if key in lyrics:
        words = lyrics[key][0]
        probabilities = lyrics[key][1]
        new_words = words + [word]
        new_probabilities = calculate_probabilities(probabilities, len(lyrics))
        lyrics[key] = [new_words, new_probabilities]
    else:
        lyrics[key] = [[word], [1]]


def is_valid_line(line):
    return not (not line or line[0] == '[')


def read_lyrics(lyrics_dict, lyrics_json):
    lyrics = lyrics_json['Lyrics'][0]
    for line in lyrics:
        if is_valid_line(line):
            tokens = line.split(' ')
            for i in range(0, len(tokens) - 1):
                add_pair(lyrics_dict, tokens[i], tokens[i + 1], even_probabilities)
                
            add_pair(lyrics_dict, tokens[-1], 'END_OF_LYRIC', even_probabilities)
                

def get_starting_word(lyrics):
    return random.choice(lyrics)


def get_random_value_from_key(lyrics, key):
    if not key in lyrics:
        return ''
    
    words = lyrics[key][0]
    probabilities = lyrics[key][1]
    
    return np.random.choice(words, p = probabilities)


def create_sentence(lyrics, starting_word = ''):
    current_word = starting_word
    if (not current_word or not current_word in lyrics):
        current_word = get_starting_word(list(lyrics.keys()))
        
    next_word = ''
    sentence = current_word
    
    while(current_word != 'end_of_lyric'):
        next_word = get_random_value_from_key(lyrics, current_word)
        sentence = f'{sentence} {next_word}'
        current_word = next_word
        
    return sentence.replace(' end_of_lyric', '')


def laplace_smoothing(lyrics):
    keys = list(lyrics.keys())
    
    for key1 in keys:
        for key2 in keys:
            add_pair(lyrics, key1, key2, weighted_probabilities)
            
        current_probabilities = lyrics[key1][1]
        lyrics[key1][1] = [i / sum(current_probabilities) for i in current_probabilities]


lyrics_directory = r'../data/output'

lyrics_dict = {}

init_word = sys.argv[1] if len(sys.argv) >= 2 else ''

use_laplace = '--laplace' in sys.argv

for filename in os.listdir(lyrics_directory):
    with open(f'{lyrics_directory}/{filename}', encoding='cp1252') as f:
        data = f.read()
        lyrics_json = json.loads(data)

        read_lyrics(lyrics_dict, lyrics_json)


if use_laplace:
    laplace_smoothing(lyrics_dict)

print(create_sentence(lyrics_dict, init_word))