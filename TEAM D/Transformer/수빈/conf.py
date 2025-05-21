"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 8
max_len = 256
d_model = 128
n_layers = 2
n_heads = 2
ffn_hidden = 256
drop_prob = 0.1

# optimizer parameter setting
init_lr = 5e-4
factor = 0.9
adam_eps = 1e-8
patience = 5
warmup = 50
epoch = 2
clip = 1.0
weight_decay = 1e-4
inf = float('inf')
