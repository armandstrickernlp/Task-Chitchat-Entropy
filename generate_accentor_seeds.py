# generate accentor seed

import json
import os
import random
import shutil
import logging
from tqdm import tqdm


def generate_seeds(seed_list): 
    for seed in seed_list: 
        logging.info('Accentor seed %d is being prepared', seed)
        random.seed(seed)
        path = './data/accentor-sgd'
        new_path = './data/accentor_seed'+str(seed)
        if not os.path.isdir(new_path):
                os.mkdir(new_path)
        for d in os.listdir(path):
            if not d.startswith('.'):
                if not os.path.isdir(new_path+'/'+d):
                    os.mkdir(new_path+'/'+d)
                logging.info(f'Split {d} is being prepared')
                for f in tqdm(os.listdir(path+'/'+d)):
                    if not f.startswith('.'):
                        if f == 'schema.json':
                            shutil.copyfile(path+'/'+d+'/'+f, new_path+'/'+d+'/'+f)
                            continue
                        with open(path+'/'+d+'/'+f, 'r') as file:
                            #print(path+'/'+d+'/'+f)
                            dialogues_n = json.load(file)
                            for n in dialogues_n:
                                for turn in n['turns']:
                                    if turn['speaker'] == 'SYSTEM':
                                        beg = 0
                                        end = 0
                                        if len(turn['beginning']) > 0:
                                            for candidate in turn['beginning']:
                                                if candidate['label'] == 'good':
                                                    beg = 1
                                        if len(turn['end']) > 0:
                                            for candidate in turn['end']:
                                                if candidate['label'] == 'good':
                                                    end = 1
                                        choice = ''
                                        if beg and not end:
                                            choice = 'beginning'
                                        elif not beg and end:
                                            choice = 'end'
                                        elif beg and end:
                                            choice = random.randint(0,1)
                                            if choice:
                                                choice = 'end'
                                            else:
                                                choice = 'beginning'
                                        else:
                                            continue
                                    
                                        good_candidates = []
                                        for candidate in turn[choice]:
                                            if candidate['label'] == 'good':
                                                good_candidates.append(candidate)
                                        select = random.randint(0, len(good_candidates)-1)
                                        chitchat = good_candidates[select]['candidate']
                                        
                                        if choice == 'beginning':
                                            turn['utterance_with_chat'] = chitchat + ' ' + turn['utterance']
                                        else :
                                            turn['utterance_with_chat'] = turn['utterance'] + ' ' + chitchat
                                        turn['added_chitchat'] = chitchat
                                
                    with open(new_path+'/'+d+'/'+f, 'w') as new:
                        json.dump(dialogues_n, new, indent=4)   
                                
            #         break
            # break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    seed_list = [1] # 5 seeds for exps
    generate_seeds(seed_list)

   