import numpy as np
from collections import Counter

"""
add_words 
load_embedding
__getitem__ -> word or index
size 
len
"""
class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3
    
    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.revered = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.revered[:]
        self.embeddings = None

    def add_words(self, words):
        "Add a new token to the vocab"
        for word in words:
            if word not in word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)
        self.word2count.update(words)
        
    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self.word2index)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims)).astype(dtype)
                        )
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings
    
    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)
    
    def __len__(self):
        return len(self.index2word)
    
    def size(self):
        return len(self.index2word)




















