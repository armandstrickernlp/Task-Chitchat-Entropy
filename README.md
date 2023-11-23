# Enhancing Task-Oriented Dialogues with Chitchat: a Comparative Study Based on Lexical Diversity and Divergence
This project compares the effect of enhancing task-oriented dialogues with different types of chitchat strategies. Metrics used are entropy-based and measure the lexical diversity and divergence brought on by the chitchat enhancements. The project relies on the following datasets: [Accentor](https://github.com/facebookresearch/accentor), [KETOD](https://github.com/facebookresearch/ketod) and [FusedChat](https://github.com/tomyoung903/FusedChat).  These datasets are all open-source and can be freely downloaded. 



## Requirements:

This project uses Python 3.9+

Create a virtual environment:

```bash
conda create -n task_chitchat_div python=3.9
```

Install the requirements:
```bas
git clone git@github.com:armandstrickernlp/Task_Chitchat_Diversity.git
cd Task_Chitchat_Diversity
pip install -r requirements.txt
```

## Compare lexical diversity and divergence
Functions for extracting task and chitchat utterances are in the `load_utts.py` script. In Accentor, chitchat is presented via several candidates for each utterance. Therefore, prior to loading the data, we randomly pick a chitchat candidate to augment each turn, when possible, using the `generate_accentor_seeds.py` script.  

The serialized extracted utterances are available in the `utt_data` repository.  

To reproduce comparison plots and results from the paper simply run the following command:

```bash
python compare_diversity_divergence.py
```

This will generate the plots in a `plots` folder.
