import json
import os
import re
from collections import Counter, defaultdict
from lexical_diversity import lex_div as ld
from nltk.util import ngrams
import numpy as np


def ngram_entropy(flat_list, n=1):
    """
    Split utterances into ngrams (n) and compute Shannon's entropy over those ngrams
    Input : flat list of utterances
    Output : entropy 
    """
    all_ngrams = []
    for utt in flat_list:
        tokenized_utt = ld.tokenize(utt)
        tokenized_utt = re.sub(r'\[ ([a-z_]+) \]', r'[\g<1>]', ' '.join(tokenized_utt)).split() #keep [price] as is vs. seperating brackets if delex
        n_grams = list(ngrams(tokenized_utt, n))
        all_ngrams += n_grams
    
    ngram_counts = Counter(all_ngrams)
    ngram_probs = {k:v/len(all_ngrams) for k, v in ngram_counts.items()}
    ngram_entropy = -np.sum([proba*np.log2(proba) for proba in ngram_probs.values()])
    return round(ngram_entropy, 3)

def get_ratios(ents1, ents2):
    # get uncertainty ratios for 2 lists of entropy measures
    return [round( 2**(ent1)/2**(ent2), 3) for ent1, ent2 in zip(ents1, ents2)]

def ngram_conditional_entropy(flat_list, n=2):
    """
    Input: flat list of utterances, same as `ngram_entropy`
    Output: conditional entropy
    """
    all_ngrams = []
    all_n_minus1_grams = []

    for utt in flat_list:
        tokenized_utt = ld.tokenize(utt)
        tokenized_utt = re.sub(r'\[ ([a-z_]+) \]', r'[\g<1>]', ' '.join(tokenized_utt)).split() #preserve delex tokens for delex version
        n_grams = list(ngrams(tokenized_utt, n))
        all_ngrams += n_grams
        n_minus1_grams = list(ngrams(tokenized_utt, n-1))
        all_n_minus1_grams += n_minus1_grams
    
    ngram_counts = Counter(all_ngrams)
    # frequ(w1,w2)/total(ngrams)

    ngram_probs = {k:v/len(all_ngrams) for k, v in ngram_counts.items()}
   
    n_minus1_counts = Counter(all_n_minus1_grams)
    # frequ(w1,w2)/frequ(w1)
    conditional_probs = {k:v/n_minus1_counts[k[:-1]] for k, v in ngram_counts.items()}
    
    # -sum (p(w1,w2) * log2(p(w2|w1))
    conditional_entropy =  -np.sum([ngram_probs[ngram] * np.log2(conditional_probs[ngram]) \
                                                                 for ngram in ngram_probs])
    
    return round(conditional_entropy,3)



def jensen_shannon_divergence(corpus1, corpus2, n=1, ind_contribs=False):
    """
    Computes divergence between 2 distributions. Similar to KL divergence but symmetric and avoids any problems if words are present in one corpus and not another.
    Can be used to find out the idividual contributions of words to the overall divergence.
    Input : - 2 flat_lists of utterance (corpus1, corpus2)
            - ngram size
            - ind_contribs: if individual contributions dict should be returned
    Output : -divergence score (0 if corpora are the same, 1 of no words in common, otherwise value between 0 and 1).
             -dict with individual contributions (from highest to lowest div score) 
    """
    #distrib_P 
    ngrams_in_P = []
    for utt in corpus1:
        tokenized_utt = ld.tokenize(utt)
        #print(' '.join(tokenized_utt))
        tokenized_utt = re.sub(r'\[ ([a-z_]+) \]', r'[\g<1>]', ' '.join(tokenized_utt)).split()
        #print(tokenized_utt)
        ngrams_utt = list(ngrams(tokenized_utt, n))
        ngrams_in_P += ngrams_utt
    distrib_P = {ngram : count/len(ngrams_in_P) for ngram, count in Counter(ngrams_in_P).items()}
    #print(len(distrib_P))
    
    # distrib_Q
    ngrams_in_Q = []
    for utt in corpus2:
        tokenized_utt = ld.tokenize(utt)
        tokenized_utt = re.sub(r'\[ ([a-z_]+) \]', r'[\g<1>]', ' '.join(tokenized_utt)).split()
        ngrams_utt = list(ngrams(tokenized_utt, n))
        ngrams_in_Q += ngrams_utt
    distrib_Q = {ngram : count/len(ngrams_in_Q) for ngram, count in Counter(ngrams_in_Q).items()}
    #print(len(distrib_Q))
    
    #distrib_M : average of distrib_P and distrib_Q
    ngrams_in_M = set(ngrams_in_P+ngrams_in_Q)
    distrib_M = defaultdict(float)
    for n in ngrams_in_M:
        if n in distrib_P:
            distrib_M[n] += distrib_P[n]
        if n in distrib_Q:
            distrib_M[n] += distrib_Q[n]
        distrib_M[n] *= 0.5     
    
    # D_js(P||Q) = 0.5*(D_kl(P||M) + D_kl(Q||M))
    KL_PM = np.sum([distrib_P[n_gram] * np.log2(distrib_P[n_gram]/distrib_M[n_gram]) for n_gram in distrib_P])
    KL_QM = np.sum([distrib_Q[n_gram] * np.log2(distrib_Q[n_gram]/distrib_M[n_gram]) for n_gram in distrib_Q])    
    JS_PQ = 0.5*(KL_PM + KL_QM)
    
    if ind_contribs:
        # D_js,i(P||Q) = mi x 0.5 x (ri x log2(ri) + (2-ri x log2(2-ri) 
        r = {ngram: distrib_P[ngram]/distrib_M[ngram] if ngram in distrib_P else 0. for ngram in distrib_M} 
        ind_contributions = {}
        for ngram in distrib_M:
            if ngram in distrib_P and r[ngram] != 2.:
                ind_contributions[ngram] = distrib_M[ngram]*0.5*(r[ngram]*np.log2(r[ngram])+(2-r[ngram])*np.log2(2-r[ngram]))
            elif ngram in distrib_P and r[ngram] == 2.:
                ind_contributions[ngram] = distrib_M[ngram]*0.5*(r[ngram]*np.log2(r[ngram]))
            else:
                ind_contributions[ngram] = distrib_M[ngram]

        # tag each contribution with corpus label
        tagged_contribution = {}
        for ngram in ind_contributions:
            if ngram in distrib_P and ngram in distrib_Q:
                if distrib_P[ngram] > distrib_Q[ngram]:
                    tagged_contribution[ngram] = (ind_contributions[ngram], "distrib_P")
                elif distrib_P[ngram] == distrib_Q[ngram]:
                    tagged_contribution[ngram] = (ind_contributions[ngram], "equal")
                else:
                    tagged_contribution[ngram] = (ind_contributions[ngram], "distrib_Q") 
            elif ngram not in distrib_P:
                tagged_contribution[ngram] = (ind_contributions[ngram], "distrib_Q")
            else:
                tagged_contribution[ngram] = (ind_contributions[ngram], "distrib_P")

        sorted_tagged = dict(sorted(tagged_contribution.items(), key=lambda item:item[1][0], reverse=True))
        return round(JS_PQ,3), sorted_tagged
    
    else:
        return round(JS_PQ,3)
