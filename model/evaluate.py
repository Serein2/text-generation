import os
import sys
import pathlib
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import collate_fn
import config



def evaluate(model, val_data, epoch):

    print('validating')
    DEVICE = torch.device('cuda' if config.is_cuda else 'cpu')
    batch_losses = []
    with torch.no_grad():
        for batch, data in enumerate(data):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            
            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches)

            batch_losses.append(loss.item())

    return np.mean(batch_losses)



