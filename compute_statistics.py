import numpy as np
import json
import pickle as pkl
from collections import defaultdict

import seaborn as sns
import re
import pickle


def compute_length_statistics(target_set, file_path):
    with open(file_path, 'r') as f:
        generated_conversations = f.readlines()

    turns = []
    for conv, target_item in list(zip(generated_conversations, target_set)):
        conv = json.loads(conv)
        if 'conv' in conv:
            conv = conv['conv']
        for idx, utt in enumerate(conv):
            if utt['role'] == 'system' and target_item.lower() in utt['content'].lower():
                # successful before the k-th turn
                turns.append(idx)

    return turns


def get_item_set(target_set_path):
    with open(target_set_path, 'rb') as f:
        target_set = pkl.load(f)

    target_item_names = []
    for item in target_set:
        target_item_names.append(item['topic'])

    return target_item_names


def compute_objective_SR(file_path, target_set):
    with open(file_path, 'r') as f:
        generated_conversations = f.readlines()

    sr = []
    for conv, target_item in list(zip(generated_conversations, target_set)):
        # instance = json.loads(line)
        conv = json.loads(conv)
        # conv = instance['conv']
        check = False
        # target_item = re.sub(r'\(\d+\)', '', target_item)
        for idx, utt in enumerate(conv):
            if utt['role'] == 'system' and target_item.lower().strip() in utt['content'].lower():
                # successful before the k-th turn
                check = True
        if check == True:
            sr.append(1)
        else:
            sr.append(0)

    return sum(sr) / len(sr)


if __name__ == '__main__':
    target_set_path = 'target_set_full_1/target.pkl'
    generated_convs_path = 'rtcp/target_set_1/mcts/generated_conversations.txt'

    o_sr = compute_objective_SR(generated_convs_path, get_item_set(target_set_path))
    print(o_sr)
