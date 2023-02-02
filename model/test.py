import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from utils import repalce_oovs

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config

class Encoder(nn.Module):
    def __init__(self,
                vocab_size,
                embed_size,
                hidden_size,
                rnn_drop: float = 0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)
        
    
    def forward(self, x):
        """
        x(Tensor): batch_size, seq_len
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embed)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.Wh = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.Ws = nn.Linear(2 * hidden_size, 2 * hidden_size)

        if config.coverage:
            Wc = nn.Linear(1, 2 * hidden_size, bias=False)

        self.v = nn.Linear(2 * hidden_size, 1, bias=False)
    
    
    def forward(self, decoder_states, encoder_output, x_padding_masks, coverage_vector=None):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        h_dec, c_dec = decoder_states
        
        # [batch_size, 1, 2 * hidden_size]
        s_t = torch.cat([h_dec, c_dec], dim=-1).transpose(0, 1)
        # [batch_size, seq_len, 2 * hidden_size]
        s_t = s_t.expand_as(encoder_output).contiguous()

        decoder_features = self.Ws(s_t)
        encoder_features = self.Wh(encoder_output)
        
        atten_inputs = encoder_features + decoder_features

        if config.coverage:
            coverage_feature = self.Wc(coverage_vector.unsqueeze(2))
            atten_inputs += coverage_feature
        
        score = self.v(atten_inputs)
        # [batch_size, seq_length]
        atten_weights = F.softmax(score, dim=1).squeeze(2)
        atten_weights = atten_weights * x_padding_masks
        atten_weights = torch.norm(atten_weights, p=-1)
        
        # [batch_size, 1, 2 * hidden_size]
        context_vector = torch.bmm(atten_weights.unsqueeze(1), decoder_output).squeeze(1)
        if config.coverage:
            coverage_vector += atten_weights
        
        return context_vector, atten_weights, coverage_vector


    class Decoder(nn.Module):
        def __init__(self, 
                    vocab_size,
                    hidden_size,
                    embedd_size):
            super(Decoder, self)
            self.hidden_size = hidden_size
            self.vocab_size = self.vocab_size
            self.DEVICE = torch.device('cuda') if config.is_cuda else torch.device("cpu")

            self.embedding = nn.Embedding(vocab_size, embed_size)

            self.lstm = nn.LSTM(embedd_size, hidden_size, batch_first=True)

            self.W1 = nn.Linear(3 * hidden_size, hidden_size)
            self.W2 = nn.Linear(hidden_size, vocab_size)

            if config.pointer:
                self.w_gen = nn.Linear(4 * hidden_size + embedd_size, out_features)
        
        def forward(self, decoder_states, context_vector, x_t):
            decoder_emb = self.embedding(x_t)

            decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)
            

            concat_vector = torch.cat([decoder_output.view(-1, config.hidden_size), context_vector], dim=1)

            p_vocab = F.softmax(self.W2(self.W1(concat_vector)), dim=1)

            h_dec, c_dec = decoder_states
            s_t = torch.cat([h_dec, c_dec], dim=-1).squeeze(0)
            
            p_gen = None
            if config.pointer:
                x_gen = torch.cat([
                    s_t, context_vector,decoder_emb.squeeze(1)
                ], dim=-1)
            p_gen = F.sigmoid(w_gen(x_gen))

            return p_vocab, p_gen, decoder_states

        
    class ReduceState(nn.Module):
        def __init__(self):
            super(ReduceState, self)

        def forward(self, encoder_states):
            enc_c, enc_d = encoder_states

            reduce_c = torch.sum(enc, 0, keepdim=True)
            reduce_d = torch.sum(enc_d, 0, keepdim=True)

            return reduce_c, reduce_d
    
    class PGN(nn.Module):
        def __init__(self, v):
            self.v = v
            self.attention = Attention(config.hidden_size)
            self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
            self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)

            self.reduce_state = ReduceState()

        def get_final_distribution(self, x, p_gen, p_vocab, atten_weights, max_oov):
            """Calculate the final distribution for the model.

        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.

        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """
            if not config.pointer:
                return p_vocab
            batch_size = x.size()[0]

            # clip the probalilties
            p_gen = torch.clamp(p_gen, 0.001, 0.999)

            p_vocab_weighted = p_gen * p_vocab
            # Get the weighted probalities
            # 注意这一步
            attention_weighted = (1 - p_gen) * attention_weights

            # extended_size = len(self.v) + max_oovs
            extension = torch.zeros(batch_size, max_oov).float().to(self.DEVICE)
            # (batch_size, extend_vocab)
            p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

            # Add the attention weights to the corresponding vocab positions.
            # Refer to equation (9).
            final_distribution = \
                p_vocab_extended.scatter_add(dim=1,
                                                index=x, src=attention_weighted)

            return final_distribution




        def forward(self, x, x_len, y, len_oovs, batch, num_batches):
            """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (Tensor):
                The numbers of out-of-vocabulary words for samples in this batch.
            batch (int): The number of the current batch.
            num_batches(int): Number of batches in the epoch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """
            x_copy = repalce_oovs(x, self.v)
            x_padding_masks = torch.ne(x, 0).byte().float()

            encoder_output, encoder_states = self.encoder(x_copy)
            decoder_states = self.reduce_state(encoder_states)

            coverage_vector = None
            if config.coverage:
                coverage_vector = torch.zeros(x_copy.shape).to(DEVICE)
            
            step_losses = []

            for t in range(y.shape[1] - 1):
                x_t = y[:, t+1]
                x_t = repalce_oovs(x_t, vocab)

                y_t = y[:, t+1]

                context_vector, atten_weights, coverage_vector = \
                    self.attention(decoder_states, encoder_output,  x_padding_masks, coverage_vector)
                
                p_vocab, p_gen, decoder_states =  self.decoder(decoder_states, \
                    context_vector, x_t.unsqueeze(1))
                
                final_dist = \
                    self.get_final_distribution(x_t, p_gen, p_vocab, atten_weights, len_oovs)
                
                if not config.pointer:
                    y_t = repalce_oovs(y_t, self.v)
                
                # [batch_size, 1]
                target_probs = torch.gather(final_dist, 1,  y_t.unsqueeze(1))
                # [batch_size]
                target_probs = target_probs.squeeze(1)

                # [batch_size]
                loss = -torch.log(target_probs + config.eps)
                
                mask = torch.ne(y_t, 0).float().to(DEVICE)
                
                if config.coverage:
                    ct_min = torch.min(coverage_vector, atten_weights)
                    cov_loss = cv_min.sum(dim=1)
                    loss  = loss + config.LAMBDA * cov_loss
                loss = loss * mask
                step_losses.append(loss)
            sample_losses = torch.sum(torch.stack(step_losses, dim=1), dim=1)
            batch_seq_len = torch.ne(y, 0).float().sum(dim=1).to(DEVICE)

            batch_loss = torch.mean(sample_losses / batch_seq_len)
        
            return batch_loss




            




            









            


        



            

            


            



            






        



        
            

        
        
        





        







        
