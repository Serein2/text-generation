
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