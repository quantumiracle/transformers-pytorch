import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from torch.optim import Adam
from adam_atan2_pytorch import AdoptAtan2

from models.skip_transformer import (
    SkipTransformer,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# constants

SAVE_DIR = './saved_models'
SAVE_FILENAME = 'transformer.pt'
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
USE_FAST_INFERENCE = False
SEQ_LEN = 512
DYNAMIC_TANH = False

# experiment related

PROJECT_NAME = 'transformer'
RUN_NAME = f'skiptransformer'
WANDB_ONLINE = True # turn this on to pipe experiment to cloud

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()



# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


# initialize transformer
model = SkipTransformer(
    num_tokens = 256,
    N = 3,  # skip size
    dim = 384,
    depth = 8,
    heads = 8,
    dim_head = 64,
    mlp_dim = 512,
    dynamic_tanh = DYNAMIC_TANH,
    ).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# training

os.makedirs(SAVE_DIR, exist_ok=True)
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        loss.backward()

    # print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = loss.item()), step = i)
    
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            print(f'validation loss: {loss.item()}')
            wandb.log(dict(val_loss = loss.item()), step = i)
    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        # get random index from val_dataset
        random_index = random.randint(0, len(val_dataset) - 1)
        inp = val_dataset[random_index][:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
        output_str = decode_tokens(sample[0])
        # also print ground truth
        print('output_str: ', output_str)
        print('ground_truth: ', decode_tokens(val_dataset[random_index][PRIME_LENGTH:GENERATE_LENGTH]))

model_save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
torch.save(model.state_dict(), model_save_path) 