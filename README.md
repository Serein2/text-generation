# text-generation
京东电商文本生成

PGN模型
Encoder          x    output, hidden
attenion         decoder_states, encoder_output, coverage_vector 
reduce
Decoder          x, decoder_states, coverage_vector, encoder_output, attention_weights
return p_gen, p_vocab, coverage_vector

PGN(v):
将oov进行替换

定义get函数
获得p_vocab * p_gen

获得attention_weight * (1 - p_gen)

return final_dist

for t in range(len(y)):

    产生loss， 进行mask
    同时加上coverage
    
return loss

