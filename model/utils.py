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