import pickle
import os
import sys
import pathlib

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from dataset import PairDataset, SimpleDataset, collate_fn
from model import Encoder, Attention, Decoder
import config

