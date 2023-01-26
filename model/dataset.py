import sys
import os
import pathlib
from typing import Callable
from collections import Counter
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from vocab import Vocab
import config
from utils import count_words, simple_tokenizer, \
                    source2ids, abstract2ids, sort_batch_by_len
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

        with open(filename, 'rt', encoding='utf8') as f:
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
    

class SimpleDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.src_sents = [x[0] for x in pairs]
        self.tgt_sents = [x[1] for x in pairs]
        self.vocab = vocab
        self._len = len(pairs)
        
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, item):
        x, oov = source2ids(self.src_sents[item], self.vocab)
        return{
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),

            'y': [self.vocab.SOS] +
            abstract2ids(self.tgt_sents[item],
            self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[item]),
            'y_len': len(self.tgt_sents[item])
        }
    
def collate_fn(batch):
    def padding(indice, max_length, pad_idx = 0):
        pad_indice = torch.tensor([x + [pad_idx] * max(0, max_length - len(x))
        for x in indice])
        return pad_indice

    data_batch = sort_batch_by_len(batch)
    x = data_batch['x']
    x_max_length = max([len(t) for t in x])
    y = data_batch['y']
    y_max_length = max([len(t) for t in y])

    OOV = data_batch['OOV']
    len_OOV = torch.tensor(data_batch['len_OOV'])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV











