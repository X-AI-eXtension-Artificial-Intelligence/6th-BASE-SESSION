""" 
하이퍼파라미터, 환경변수 정의 
""" 


import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # model parameter setting
# batch_size = 128
# max_len = 256
# d_model = 512
# n_layers = 6
# n_heads = 8
# ffn_hidden = 2048
# drop_prob = 0.1

# # optimizer parameter setting
# init_lr = 1e-5
# factor = 0.9
# adam_eps = 5e-9
# patience = 10
# warmup = 100
# epoch = 1000
# clip = 1.0
# weight_decay = 5e-4
# inf = float('inf')



batch_size = 4
max_len = 128  # Sequence 길이도 줄여야 메모리 사용량 감소
d_model = 256
n_layers = 2
n_heads = 4
ffn_hidden = 512
drop_prob = 0.1

init_lr = 1e-4
factor = 0.9
adam_eps = 1e-8
patience = 5
warmup = 10
epoch = 50
clip = 1.0
weight_decay = 1e-4
inf = float('inf')
