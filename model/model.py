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
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)

    def forward(self, x):
        """
         x (Tensor): The input samples as shape (batch_size, seq_len)
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()

        self.Wh = nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False)
        self.Ws = nn.Linear(2 * hidden_units, 2 * hidden_units)
        

        self.v = nn.Linear(2 * hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks):
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
        # Concatenate h and c
        h_dec, c_dec = decoder_states
        print(h_dec.shape, c_dec.shape)
        # (1, batch_size, , 2 * hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # (batch_size, 1, 2 * hidden_units)
        s_t = s_t.transpose(0, 1)
        print(encoder_output.shape, s_t.shape)
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        encoder_features = self.Wh(encoder_output.contiguous())
        decoder_features = self.Ws(s_t)
        att_inputs = encoder_features + decoder_features
        score = self.v(torch.tanh(att_inputs))

        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        # normalize attention weights
        # batch_size, seq_length
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        #(batch_size, 1, 2 * hidden_units)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)

        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if config.is_cuda else torch.device("cpu")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.

        Args:
            x_t (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
            p_gen (Tensor):
                The generation probabilities of shape (batch_size, 1).
        """
        # (batch_size,seq_length, embed_size)
        decoder_emb = self.embedding(x_t)
        # (batch_size, seq_length, hidden_size)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # (batch_size * seq_length, hidden_size)
        # 此处的seq_length = 1
        decoder_output = decoder_output.view(-1, config.hidden_size)

        # concatenate context vector and decoder state
        concat_vector = torch.cat([decoder_output, concat_vector], dim=-1)
        # calculate vocablary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2 * hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        p_gen = None
        if config.pointer:
            x_gen = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ], dim=-1)

            p_gen = torch.sigmoid(self.w_gen(x_gen))


        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.

        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).

        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)

        return (h_reduced, c_reduced)



class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v), 
            config.embed_size, 
            config.hidden_size
            )
        self.decoder = Decoder(
            len(v), 
            config.embed_size,
            config.hidden_size)
        
        self.reduce_state = ReduceState()

    def load_model(self):
        pass


    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights,
                               max_oov):
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

        # clip the probalilities
        p_gen = torch.clamp(p_gen, 0.001, 0.999)

        # Get the weighted probalities
        # Refer to equation (9).
        p_vocab_weighted = p_gen * p_vocab
        #(batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        #extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        #(batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # Add the attention weights to the corresponding vocab positions.
        # Refer to equation (9).
        final_distribution = \
            p_vocab_extended.scatter_add(dim=1,
                                            index=1, src=attention_weighted)
        
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
        # Reduce encoder hidden states
        decoder_states = self.reduce_state(encoder_states)
        # Initialize coverage vector
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)
        step_losses = []
        # Calculate loss for every step
        for t in range(y.shape[1] - 1):
            # Do teacher forcing
            x_t = y[:, t]
            x_t = repalce_oovs(x_t, self.v)

            y_t = y[:, t+1]
            # Get context vector from attention network
            context_vector, attention_weights, coverage_vector = \
                self.attention(decoder_states, 
                               encoder_output, 
                               x_padding_masks,
                               coverage_vector)
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states, p_gen = self.decoder(
                x_t.unsqueeze(1),
                decoder_states,
                context_vector)
            final_dist = self.get_final_distribution(x,
                                                     p_gen, 
                                                     p_vocab, 
                                                     attention_weights, 
                                                     torch.max(len_oovs))
            if not config.pointer:
                y_t = repalce_oovs(y_t, self.v)
            
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(y_t, 0).byte().float()
            # Do smoothing to prevent getting NaN loss because of log(0).
            # bs 
            loss = -torch.log(target_probs + config.eps)

            if config.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)

            loss = loss * mask

            step_losses.append(loss)
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)

        seq_len_mask = torch.ne(y, 0).byte().float()
        # batch_size
        batch_seq_len = torch.sum(seq_len_mask, dim=1)
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
















