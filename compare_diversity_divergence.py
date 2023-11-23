import pickle
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from load_utts import load_fusedchat, load_ketod, load_accentor, load_bst
from metric_utils import ngram_entropy, get_ratios, ngram_conditional_entropy, jensen_shannon_divergence



logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s')


logging.info('Loading data')
# mwoz_lex, fchat_lex, fchat_chitchat_only = load_fusedchat('data/FusedChat/appended_lexicalized.json', 'data/FusedChat/prepended_lexicalized.json')
# task_lex, enriched_lex, knowledge_only = load_ketod('data/KETOD')
# sgd_lex, acc_lex, chitchat_only = load_accentor('data/accentor_seed0')
# BST = load_bst('data/blended_skill_talk')

# load from serialized files directly
path = 'utt_data'
with open(os.path.join(path, 'mwoz_lex.pkl'), 'rb') as f:
    mwoz_lex = pickle.load(f)
with open(os.path.join(path, 'fchat_lex.pkl'), 'rb') as f:
    fchat_lex = pickle.load(f)
with open(os.path.join(path, 'fchat_chitchat_only.pkl'), 'rb') as f:
    fchat_chitchat_only = pickle.load(f)
with open(os.path.join(path, 'task_lex.pkl'), 'rb') as f:
    task_lex = pickle.load(f)
with open(os.path.join(path, 'enriched_lex.pkl'), 'rb') as f:
    enriched_lex = pickle.load(f)
with open(os.path.join(path, 'knowledge_only.pkl'), 'rb') as f:
    knowledge_only = pickle.load(f)
with open(os.path.join(path, 'sgd_lex.pkl'), 'rb') as f:
    sgd_lex = pickle.load(f)
with open(os.path.join(path, 'acc_lex.pkl'), 'rb') as f:
    acc_lex = pickle.load(f)
with open(os.path.join(path, 'chitchat_only.pkl'), 'rb') as f:
    chitchat_only = pickle.load(f)
with open(os.path.join(path, 'BST.pkl'), 'rb') as f:
    BST = pickle.load(f)


# make plots directory
os.makedirs('plots', exist_ok=True)

logging.info('Making entropy plots')
task_lex_list, enriched_lex_list = [], []
mwoz_lex_list, fchat_lex_list = [], []
sgd_lex_list, acc_lex_list = [], []
BST_list = []

for n in range(1,4):
    task_lex_list.append(ngram_entropy(task_lex, n=n))
    enriched_lex_list.append(ngram_entropy(enriched_lex, n=n))

    mwoz_lex_list.append(ngram_entropy(mwoz_lex, n=n))
    fchat_lex_list.append(ngram_entropy(fchat_lex, n=n))

    sgd_lex_list.append(ngram_entropy(sgd_lex, n=n))
    acc_lex_list.append(ngram_entropy(acc_lex, n=n))

    BST_list.append(ngram_entropy(BST, n))



# Get plots
# Define the bar positions
x = np.arange(len(task_lex_list))
total_bars = 7
bar_width = 0.1

plt.figure(figsize=(10, 6))

# Create the bar chart
plt.bar(x - 3 * bar_width, task_lex_list, width=bar_width, color='steelblue', edgecolor='black', label='KETOD task resps.', hatch='-')
plt.bar(x - 2 * bar_width, enriched_lex_list, width=bar_width, color='skyblue', edgecolor='black', label='KETOD aug. resps.', hatch='-')
plt.bar(x - bar_width, mwoz_lex_list, width=bar_width, color='forestgreen', edgecolor='black', label='Fchat task resps.', hatch='/')
plt.bar(x, fchat_lex_list, width=bar_width, color='limegreen', edgecolor='black', label='Fchat aug. reps.', hatch='/')
plt.bar(x + bar_width, sgd_lex_list, width=bar_width, color='purple', edgecolor='black', label='Acc. task resps.', hatch='o')
plt.bar(x + 2 * bar_width, acc_lex_list, width=bar_width, color='orchid', edgecolor='black', label='Acc. aug. resps.', hatch='o')
plt.bar(x + 3 * bar_width, BST_list, width=bar_width, alpha=0.5, color='orange', edgecolor='black', label='BST resps.', hatch='')

# Add labels, title, and legend
plt.xlabel('N-gram Length')
plt.ylabel('Entropy')
plt.title('System Response Entropy')
plt.xticks(x, ['1', '2', '3'])
plt.legend()

plt.ylim(6, 18)

plt.savefig('plots/entropy.pdf', bbox_inches='tight')


# entropy ratios
ratio_fchat_lex = get_ratios(fchat_lex_list, mwoz_lex_list)
ratio_acc_lex = get_ratios(acc_lex_list, sgd_lex_list)
ratio_ketod_lex = get_ratios(enriched_lex_list, task_lex_list)

# plot the ratios
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(ratio_ketod_lex, label='KETOD', color='steelblue', linestyle='--')
ax.plot(ratio_fchat_lex, label='Fchat', color='forestgreen', linestyle='-.')
#ax.plot(ratio_fchat_delex, label='FCHAT delex', color='limegreen')
ax.plot(ratio_acc_lex, label='Acc.', color='purple', linestyle='-')
#ax.plot(ratio_acc_delex, label='ACC delex', color='orchid')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['1', '2', '3'])
ax.set_ylabel('Uncertainty ratio')
ax.set_xlabel('N-gram length')
# ax.set_title('Uncertainty Ratio for Entropy')

ax.legend()

plt.savefig('plots/ratio_entropy.pdf', bbox_inches='tight')



########################## conditional entropy
logging.info('Making conditional entropy plots')
cond_task_lex_list, cond_enriched_lex_list = [], []
cond_mwoz_lex_list, cond_fchat_lex_list = [], []
cond_sgd_lex_list, cond_acc_lex_list = [], []
cond_BST_list = []

for n in range(2,5):
    cond_task_lex_list.append(ngram_conditional_entropy(task_lex, n))
    cond_enriched_lex_list.append(ngram_conditional_entropy(enriched_lex, n))

    cond_mwoz_lex_list.append(ngram_conditional_entropy(mwoz_lex, n))
    cond_fchat_lex_list.append(ngram_conditional_entropy(fchat_lex, n))

    cond_sgd_lex_list.append(ngram_conditional_entropy(sgd_lex, n))
    cond_acc_lex_list.append(ngram_conditional_entropy(acc_lex, n))

    cond_BST_list.append(ngram_conditional_entropy(BST, n=n))

ratio_cond_ketod = get_ratios(cond_enriched_lex_list, cond_task_lex_list)
ratio_cond_fchat_lex = get_ratios(cond_fchat_lex_list, cond_mwoz_lex_list)
ratio_cond_acc_lex = get_ratios(cond_acc_lex_list, cond_sgd_lex_list)

# Plot conditional Entropies
x = np.arange(len(task_lex_list))
total_bars = 7
bar_width = 0.1

plt.figure(figsize=(10, 6))

# Create the bar chart
plt.bar(x - 3 * bar_width, cond_task_lex_list, width=bar_width, color='steelblue', edgecolor='black', label='KETOD task resps.', hatch='-')
plt.bar(x - 2 * bar_width, cond_enriched_lex_list, width=bar_width, color='skyblue', edgecolor='black', label='KETOD aug. resps.', hatch='-')
plt.bar(x - bar_width, cond_mwoz_lex_list, width=bar_width, color='forestgreen', edgecolor='black', label='Fchat task resps.', hatch='/')
plt.bar(x, cond_fchat_lex_list, width=bar_width, color='limegreen', edgecolor='black', label='Fchat aug. reps.', hatch='/')
plt.bar(x + bar_width, cond_sgd_lex_list, width=bar_width, color='purple', edgecolor='black', label='Acc. task resps.', hatch='o')
plt.bar(x + 2 * bar_width, cond_acc_lex_list, width=bar_width, color='orchid', edgecolor='black',label='Acc. aug. resps.', hatch='o')
plt.bar(x + 3 * bar_width, cond_BST_list, width=bar_width, color='orange', edgecolor='black', alpha=0.5, label='BST resps.')


# Add labels, title, and legend
plt.xlabel('Context Length (n-grams)')
plt.ylabel('Conditional Entropy')
plt.title('System Response Conditional Entropy')
plt.xticks(x + 2.5 * bar_width, ['1', '2', '3'])
plt.legend()

# Display the chart
plt.savefig('plots/cond_entropy.pdf', bbox_inches='tight')



# plot ratios
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ratio_cond_ketod, label='KETOD', color='steelblue', linestyle='--')
ax.plot(ratio_cond_fchat_lex, label='Fchat', color='forestgreen', linestyle='-.')
#ax.plot(ratio_cond_fchat_delex, label='FCHAT delex', color='limegreen')
ax.plot(ratio_cond_acc_lex, label='Acc.', color='purple')
#ax.plot(ratio_cond_acc_delex, label='ACC delex', color='orchid')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['1', '2', '3'])
ax.set_ylabel('Uncertainty ratio')
ax.set_xlabel('Context length (n-grams)')
# ax.set_title('Uncertainty Ratio for Conditional Entropy')

ax.legend()

plt.savefig('plots/ratio_cond.pdf', bbox_inches='tight')



################ Divergence
logging.info('Making divergence plots')
# KETOD
div_task_knowledge, contribs_task_knowledge = jensen_shannon_divergence(task_lex, knowledge_only, n=1, ind_contribs=True) 
div_task_BST, contribs_task_BST = jensen_shannon_divergence(task_lex, BST, n=1, ind_contribs=True)
div_BST_knowledge, contribs_BST_knowledge = jensen_shannon_divergence(BST, knowledge_only, n=1, ind_contribs=True)

# FusedChat
div_mwoz_chitchat, contribs_mwoz_chitchat = jensen_shannon_divergence(mwoz_lex, fchat_chitchat_only, n=1, ind_contribs=True)
div_mwoz_BST, contribs_mwoz_BST = jensen_shannon_divergence(mwoz_lex, BST, n=1, ind_contribs=True)
div_BST_fchatChitchat, contribs_BST_fchatChitchat = jensen_shannon_divergence(BST, fchat_chitchat_only, n=1, ind_contribs=True)

# Accentor
div_sgd_chitchat, contribs_sgd_chitchat = jensen_shannon_divergence(sgd_lex, chitchat_only, n=1, ind_contribs=True)
div_sgd_BST, contribs_sgd_BST = jensen_shannon_divergence(sgd_lex, BST, n=1, ind_contribs=True)
div_BST_chitchat, contribs_BST_chitchat = jensen_shannon_divergence(BST, chitchat_only, n=1, ind_contribs=True)


# plotting
def XPQ(contrib, topk=10):
    X, P, Q = [], [], []     
    for tup in list(contrib.items())[:topk][::-1]:
        X.append(tup[0][0])
        if tup[1][1] == 'distrib_P':
            P.append(tup[1][0])
            Q.append(0.)
        else:
            P.append(0)
            Q.append(tup[1][0])
    Q = np.array(Q)
    return X,P,Q

def plot_3divs_back2back(contribs1, contribs2, contribs3, name1, name2, name3, title="Title", rot=0):
    X1, P1, Q1 = XPQ(contribs1, topk=20)
    X2, P2, Q2 = XPQ(contribs2, topk=20)
    X3, P3, Q3 = XPQ(contribs3, topk=20)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10.2, 5), constrained_layout=True)
    ax1.barh(X1, P1)
    ax1.barh(X1, -Q1)
    ax1.set_title(name1)
    ax1.grid()
    ax1.tick_params(axis='x', labelsize=9)
    
    ax2.barh(X2, P2)
    ax2.barh(X2, -Q2)
    ax2.set_title(name2)
    ax2.grid()
    ax2.tick_params(axis='x', labelsize=9)
    
    ax3.barh(X3, P3)
    ax3.barh(X3, -Q3)
    ax3.set_title(name3)
    ax3.grid()
    ax3.tick_params(axis='x', labelsize=8)
    
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=rot)

    fig.suptitle(title, fontsize=14)
    fig.savefig(f"plots/Divergences_{title}.pdf", format='pdf', bbox_inches='tight')


plot_3divs_back2back(contribs_sgd_chitchat,
                    contribs_sgd_BST,
                    contribs_BST_chitchat,
                    "a) Chitchat vs. Task",
                    "b) BST vs. Task",
                    "c) Chitchat vs. BST",
                    title="Accentor",
                    rot=45)
                
plot_3divs_back2back(contribs_mwoz_chitchat,
                    contribs_mwoz_BST,
                    contribs_BST_fchatChitchat,
                    "a) Chitchat vs. Task",
                    "b) BST vs. Task",
                    "c) Chitchat vs. BST",
                    title="FusedChat")

plot_3divs_back2back(contribs_mwoz_chitchat,
                     contribs_task_BST, 
                     contribs_BST_knowledge,
                     "a) Chitchat vs. Task", 
                     "b) BST vs. Task", 
                     "c) Chitchat vs. BST",
                     title="KETOD")