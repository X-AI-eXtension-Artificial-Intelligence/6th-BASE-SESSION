"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128  # 배치크기:128
max_len = 256  # 최대시퀀스길이:256
d_model = 512  # 임베딩차원:512
n_layers = 6  # 레이어수:6
n_heads = 8  # 어텐션헤드수:8
ffn_hidden = 2048  # FFN은닉크기:2048
drop_prob = 0.1  # 드롭아웃확률:0.1

# optimizer parameter setting
init_lr = 1e-5  # 초기학습률:1e-5
factor = 0.9  # lr감소비율:0.9
adam_eps = 5e-9  # Adamepsilon:5e-9
patience = 10  # lr감소대기(에포크):10
warmup = 100  # 워밍업스텝:100
epoch = 1000  # 전체에포크:1000
clip = 1.0  # 그래디언트클리핑:1.0
weight_decay = 5e-4  # 가중치감쇠:5e-4
inf = float('inf')  # 무한대:inf
