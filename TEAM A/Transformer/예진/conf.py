"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# 파라미터 세팅

# GPU device setting 이었는데
#  PyTorch 버전이 서버 GPU와 호환되지 않아서 에러 -> CPU로 수정
device = torch.device("cpu")


# model parameter setting
batch_size = 16   # 메모리 문제로 기존 128 -> 16 으로 줄여줌 
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')