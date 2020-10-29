# cse-496-nlp
Final project for team Google Translate 2.0 from class SPECIAL TOPICS CSCE496 SEC 701 Fall 2020.

Lyric generation in the style of Nickelback songs.

## Dataset
Data is compiled by scraping Genius.com for Nickelback songs.
The script `data/genius_lyrics.py` accepts URLs as command line arguments and outputs the scraped lyrics in the `data/output/` directory.
Each set of lyrics is stored as a JSON file containing the lyrics as well as some metadata from Genius.com.

## Implementation
Two methods are implemented to generate lyrics in the style of Nickelback:

1. Markov processes
2. Long short-term memory (LSTM)

### Markov Processes
Implementations for Markov processes exist in the `markov_processes/` directory.

### Long Short-Term Memory
Implementations for LSTM exist in the `lstm/` directory.