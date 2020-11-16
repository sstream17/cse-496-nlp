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
The environment `project` defined by `lstm/environment.yaml` must be created before running the script. 
The Python script can be run on HCC Crane from the directory `lstm/` with the command `sbatch project.slurm lstm.py`.
  The SLURM file `project.slurm` requests 48 GB. This can easily be changed to 32 GB by changing line 3: `#SBATCH --mem=48G` to `#SBATCH --mem=32G`. The script has been tested successfully with 32 GB for 300 epochs.
  
  The implementation with GloVe weights requires downloading the weights from the [Stanford NLP group](https://nlp.stanford.edu/projects/glove/).
  The attention layer implementation is from [Marco Cerliani](https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137) on Stack Overflow.
