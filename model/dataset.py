import sys
import os
import pathlib
from typing import Callable

import torch
from torch.utils.data import Dataset
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from vocab import Vocab
import config
from utils import count_words
"""
# pairDataset 
init src and tgt and pairs
bulid_vocab

"""

class PairDataset(object):
    # 使用tokenize进行分词
    def __init__(self,
                filename,
                tokenize: Callable = simple_tokenizer,
                max_src_len: int = None,
                max_tgt_len: int = None,
                truncate_src: bool = False,
                truncate_tgt: bool = False):
        print("Reading dataset %s..." % filename, end=' ', flush=None)
        self.filename  = filename
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                pair = line.strip().split("<sep>")
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    print(line)
                    continue
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))
        print("%d paris." % len(self.pairs))

    def bulid_vocab(self, embed_file: str=None) -> Vocab:
        word_counts = Counter()
        """
        count words and build vocab
        """
        count_words(word_counts, [src + tgr for src, tgr in self.pairs])
        vocab = Vocab()

        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words(word)
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            print("%d pre-trained embeddings loaded." % count)
        
        return vocab





