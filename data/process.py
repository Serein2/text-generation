import sys
import os
import pathlib

import json
import jieba

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from data_utils import write_samplesj, partition

samples = set()

json_path = os.path.join(abs_path, "../files/服饰_50k.json")
with open(json_path, 'r', encoding='utf8') as file:
    jsf = json.load(file)

for jsobj in jsf.values():
    tilte = jsobj['title']
    kb = dict(jsobj['kb']).items()
    kb_merged = ''
    for key, val in kb:
        kb_merged += key + " " + val
    
    ocr = " ".join(list(jieba.cut(jsobj['ocr'])))
    texts = []
    texts.append(title + ocr + kb_merged)
    reference = " ".join(list(jieba.cut(jsobj['reference'])))
    for text in texts:
        sample = text + "<sep>" + reference
        samples.add(sample)

write_path = os.path.join(abs_path, "../files/samples.txt")
write_samples(samples, write_path)
partition(samples)

