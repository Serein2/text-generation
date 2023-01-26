import numpy as np
import time
import heapq
import random
import sys
import pathlib

import torch

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config

def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1

def simple_tokenizer(text):
    """
    string
    """
    return text.split()


def source2ids(source_words, vocab):
    unk = vocab['<UNK>']
    ids, oovs = [], []
    for word in source_words:
        index = vocab[word]
        if index == unk:
            if word not in oovs:
                oovs.append(word)
            oov_num = oovs.index(word)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(index)

    return ids, oovs

def outputids2words(id_list, source_oovs, vocab):
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]
        except IndexError:
            assert_msg = "ERROR ID can't find"
            assert source_oovs is not None, assert_msg
            source_oov_index = i - vocab.size()
            try:
                w = source_oovs[source_oov_index]
            except ValueError:
                raise ValueError("ERROR ID can't find OOV")
        words.append(w)
    return " ".join(words)

def abstract2ids(abstract_words, vocab, source_oovs):
    unk_id = vocab.UNK
    ids = []
    for word in abstract_words:
        i = vocab[word]
        if i == unk_id:
            if word in source_oovs:
                ids.append(source_oovs.index(word) + vocab.size())
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids
        
def sort_batch_by_len(data_batch):
    res = {
        'x': [],
        'y': [],
        'x_len': [],
        'y_len': [],
        'OOV': [],
        'len_OOV':[]}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])
        sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {
            name: [_tensor[i] for i in sorted_indices]
            for name, _tensor in res.items()
    }
    return data_batch