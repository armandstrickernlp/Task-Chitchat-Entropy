# Enhancing Task-Oriented Dialogues with Chitchat: a Comparative Study Based on Lexical Diversity and Divergence
This project compares the effect of enhancing task-oriented dialogues with different types of chitchat strategies. Metrics used are entropy-based and measure the lexical diversity and divergence brought on by the chitchat enhancements. The project relies on the following datasets: [Accentor](https://github.com/facebookresearch/accentor), [KETOD](https://github.com/facebookresearch/ketod) and [FusedChat](https://github.com/tomyoung903/FusedChat).  These datasets are all open-source and can be freely downloaded.  

Accepted @ ASRU 2023. Publication to come !



## Requirements:

This project uses Python 3.9+

Create a virtual environment:

```bash
conda create -n task_chitchat_ent python=3.9
```

Install the requirements:
```bas
git clone git@github.com:armandstrickernlp/Task-Chitchat-Entropy.git
cd Task-Chitchat-Entropy
pip install -r requirements.txt
```

## Compare lexical diversity and divergence
The serialized extracted utterances from each dataset are made available in the `utt_data` repository and are directly loaded in `compare_diversity_divergence.py`.

To reproduce comparison plots and results from the paper, simply run the following command:

```bash
python compare_diversity_divergence.py
```

This will generate the plots in a `plots` directory.

Code for computing metrics is in `metric_utils.py` and code for extracting utterances is in `load_uts.py`. 
Note: In Accentor, several chitchat candidates are proposed for most utterances. Prior to extracting the Accentor utterances for analysis, we randomly pick a chitchat candidate when possible, using the `generate_accentor_seeds.py`.
