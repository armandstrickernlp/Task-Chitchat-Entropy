import json
import os

import difflib
import nltk

from nltk import word_tokenize


def load_fusedchat(app_path, prep_path):
    with open(app_path) as f:
        app = json.load(f)

    with open(prep_path) as f:
        prep = json.load(f)

    mwoz_lex = [] # no chitchat
    fchat_lex = [] # mowz + chitchat
    chitchat_only = [] 

    for subset in [app, prep]:
        for split in subset:
            for d_num in subset[split]:
                for idx, turn in enumerate(subset[split][d_num]['log']):
                    if idx % 2 == 1:
                        if turn['metadata'] == {}: # chitchat turns
                            fchat_lex.append(turn['text'])
                            chitchat_only.append(turn['text'])
                        else:
                            fchat_lex.append(turn['text'])
                            mwoz_lex.append(turn['text'])
    
    return mwoz_lex, fchat_lex, chitchat_only


def load_ketod(ketod_path):
    task_lex, enriched_lex, knowledge_only = [], [], []

    def find_non_matching_spans(s1, s2):
        "coarse-grained extraction of knowledge snippets"
        s2_tok = word_tokenize(s2)
        s1_lower = word_tokenize(s1.lower())
        s2_lower = word_tokenize(s2.lower())
        matcher = difflib.SequenceMatcher(None, s1_lower, s2_lower)
        matching_blocks = matcher.get_matching_blocks()
        
        for match in matching_blocks:
            if match.size > 0:
                s2_tok[match.b:match.b+match.size] = ['<span>'] * match.size
        
        joined_text = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(s2_tok)
        joined_text = joined_text.replace('<span> ', '').replace(' <span>', '').replace('<span>', '')
        return joined_text


    for file_name in os.listdir(ketod_path):
        if file_name.endswith('.json'):
            with open(os.path.join(ketod_path, file_name)) as f:
                split = json.load(f)
            for dial in split:
                for idx, turn in enumerate(dial["turns"]):
                    if idx % 2 == 1:
                        task_lex.append(turn["utterance"])
                        if turn['enrich'] == False:
                            enriched_lex.append(turn["utterance"])
                        elif turn['enrich'] == True:
                            enriched_lex.append(turn["enriched_utter"])
                            kg = find_non_matching_spans(turn["utterance"], turn["enriched_utter"])
                            knowledge_only.append(kg) 
    
    return task_lex, enriched_lex, knowledge_only 


def load_bst(bst_path):    
    sp2 = []
    for f in os.listdir(bst_path):
        if f.endswith('.json') :

            with open(os.path.join(bst_path, f)) as file:
                dialogues_n = json.load(file)

            for n in dialogues_n:

                sp2.append(n['guided_turker_utterance'])
                for turn in n['dialog']:
                    if turn[0] == 1:
                        sp2.append(turn[1])
    return sp2


def load_accentor(accentor_path):
      
    sgd_lex, acc_lex, chitchat_only = [], [], []

    for d in os.listdir(accentor_path):
        if not d.startswith('.'):
            for f in os.listdir(os.path.join(accentor_path, d)):
                if not f.startswith('.') and f != 'schema.json':
                    with open(os.path.join(accentor_path, d, f)) as file:
                        dialogues_n = json.load(file)
                    for n in dialogues_n:
                        for turn in n['turns']:
                            if turn['speaker'] == 'SYSTEM':
                                sgd_lex.append(turn['utterance'].strip())
                                if turn.get('utterance_with_chat') is not None:
                                    acc_lex.append(turn['utterance_with_chat'].strip())
                                    chitchat_only.append(turn['added_chitchat'].strip()) 
                                else:
                                    acc_lex.append(turn['utterance'].strip())
                            
    return sgd_lex, acc_lex, chitchat_only


if __name__ == '__main__':
    ...
     # load_fusedchat('data/FusedChat/appended_lexicalized.json', 'data/FusedChat/prepended_lexicalized.json')
    # load_ketod('data/KETOD')