import pickle
import os
import sys
import pathlib

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm
from evaluate import evaluate
from model import PGN

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import PairDataset, SimpleDataset, collate_fn
from model import Encoder, Attention, Decoder
import config

def train(dataset, val_dataset, v, start_epoch = 0):

    DEVICE = torch.device('cuda' if config.is_cuda else 'cpu')

    model = PGN(v)
    model.load_model()


    # forward
    print("loading data")
    train_data = SimpleDataset(dataset.pairs, v)
    val_data = SimpleDataset(val_data.pairs, v)

    print("initializing optimizer")

    optimizer = optim.Adam(model.parameters(), 
                            lr = config.learning_rate)
    
    train_dataloader = DataLoader(train_data, 
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    
    val_losses = np.inf
    
    
    # 
    with tqdm(total=config.epochs) as epoch_progress:
        for epoch in range(start_epoch, config.epochs):
            batch_losses = []
            num_batches = len(train_dataloader)
            with tqdm(total=num_batches//100) as batch_progress:
                for batch, data in enumerate(tqdm(train_dataloader)):
                    x, y, x_len, y_len, oov, len_oovs = data
                    assert not np.any(np.isnan(x.numpy()))
                    if config.is_cuda:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    model.train()
                    optimizer.zero_grad()
                    
                    loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches)
                    batch_losses.append(loos.item())

                    loss.backward()

                    # Do gradient clipping to prevent gradient explosion.
                    clip_grad_norm_(model.encoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(),
                                    config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(),
                                    config.max_grad_norm)

                    optimizer.step()

                    if batch % 100 == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch,
                                                   Loss=loss.item())
                        batch_progress.update()

            epoch_loss = np.mean(batch_losses)
            epoch_progress.set_description(f'Epoch {epoch}')
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()
            
            avg_val_loss = evaluate(model, val_data, epoch)
            print('training loss:{}'.format(epoch_loss),
                  'validation loss:{}'.format(avg_val_loss))


            if avg_val_loss < val_losses:
                torch.save(model.encoder, config.encoder_save_name)
                torch.save(model.decoder, config.decoder_save_name)
                torch.save(model.reduce_state, config.reduce_state_save_name)
                torch.save(model.attention, config.attention_save_name)
                with open(config.losses_path, 'wb') as f:
                    pickle.dump(val_losses, f)



if __name__ == '__main__':
    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
    dataset = PairDataset(config.data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    val_dataset = PairDataset(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset, val_dataset, vocab, start_epoch=0)






                    



