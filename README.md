# Enhancing Task-Oriented Dialogues with Chitchat: a Comparative Study Based on Lexical Diversity and Divergence

## Requirements:

This project uses Python 3.9+

Create a virtual env with conda:

```bash
conda create -n task_chitchat_div python=3.9
```

Install the requirements (inside the project folder):
```bash
git clone git@github.com
cd TaskChitchatDiv
pip install -r requirements.txt
```

## Compare lexical diversity and divergence
WE compare 
The utterances used as input
To obtain the plots from the paper simply run the following command:

```bash
python compare_diversity_divergence.py
```

This will generate the plots in the `plots` folder.


## Data
Datasets used for this study include FusedChat, Accentor and KETOD.

Once the datasets are downloaded, `generate_accentor_seeds.py` generates possible chitchat completions out of the possible candidates for each task-oriented utterance.

`load_utts.py` loads the utterances for each dataset for comparisons

