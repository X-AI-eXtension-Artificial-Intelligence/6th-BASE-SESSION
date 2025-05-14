# conf.py : 주요 하이퍼파라미터와 환경설정 정의 파일

import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128 # 배치 크기
max_len = 256 # 입력 시퀀스의 최대 길이 제한
d_model = 512 # 모델의 임베딩 차원
n_layers = 6 # Transformer의 인코더/디코더 레이어 수
n_heads = 8 # 멀티헤드 어텐션에서 사용할 어텐션 헤드 개수
ffn_hidden = 2048 # FFN 내부의 은닉층 차원
drop_prob = 0.1 # 드롭아웃 확률

# optimizer parameter setting
init_lr = 1e-5 # 초기 학습률
factor = 0.9 # 학습률 감소 시, 감쇠 계수
adam_eps = 5e-9 # Adam optimizer의 epsilon값 (수치적 안정성)
patience = 10 # 학습률 감소 또는 조기 종료를 위해 기다리는 에폭 수
warmup = 100 # 학습 초기에 학습률을 천천히 증가시키는 워밍업 스텝 수
epoch = 1000 # 학습 반복 횟수
clip = 1.0 # 역전파시, 그래디언트 값이 너무 커지는 문제를 막기 위해 그래디언트 크기를 일정 임계값 이하로 잘라내는 기법
weight_decay = 5e-4 # optimizer의 가중치 감쇠 계수 (L2 정규화)
inf = float('inf') # 무한대 -> best_loss 초기화 등에 사용
