"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# CUDA가 가능하면 GPU 사용, 아니면 CPU 사용

# ===============================
# model parameter setting
# ===============================

batch_size = 128       # 한 번에 학습할 데이터 수
max_len = 256          # 입력 시퀀스의 최대 길이
d_model = 512          # 임베딩 및 모델 차원
n_layers = 6           # 인코더/디코더 레이어 수
n_heads = 8            # Multi-Head Attention의 헤드 수
ffn_hidden = 2048      # FFN의 내부 hidden layer 크기
drop_prob = 0.1        # 드롭아웃 확률

# ===============================
# optimizer parameter setting
# ===============================

init_lr = 1e-5         # 초기 학습률
factor = 0.9           # 학습률 감소 계수 (lr_scheduler 관련)
adam_eps = 5e-9        # Adam 옵티마이저 안정화를 위한 epsilon 값
patience = 10          # early stopping 기준 (개선 없을 경우 몇 epoch까지 기다릴지)
warmup = 100           # learning rate warm-up 단계 수
epoch = 1000           # 전체 학습 epoch 수
clip = 1.0             # gradient clipping 임계값 (폭주 방지)
weight_decay = 5e-4    # L2 regularization을 위한 weight decay 계수
inf = float('inf')     # 무한대 값 (loss 비교 등에서 사용)
