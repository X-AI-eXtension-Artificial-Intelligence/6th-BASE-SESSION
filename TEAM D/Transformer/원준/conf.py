import torch

# 디바이스 설정 (GPU가 있으면 CUDA, 없으면 CPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 하이퍼파라미터 설정
batch_size = 128       
max_len = 256          # 입력 시퀀스의 최대 길이
d_model = 512          # 토큰 임베딩 및 모델 차원 수 (Transformer 핵심 차원) 한 단어가 이 차원 
n_layers = 6           # 인코더/디코더 레이어 수
n_heads = 8            # 멀티헤드 어텐션에서의 헤드 개수
ffn_hidden = 2048      # Position-wise FFN의 은닉층 크기
drop_prob = 0.1        # 드롭아웃 확률 (과적합 방지)

# 옵티마이저 및 학습 관련 설정
init_lr = 1e-5         # 초기 학습률
factor = 0.9           # learning rate decay 계수 (scheduler 등에서 사용)
adam_eps = 5e-9        # Adam 옵티마이저에서 작은 epsilon 값 (수치 안정성 확보)
patience = 10          # validation loss가 개선되지 않을 때 기다릴 최대 epoch 수
# 검증 손실(val loss)이 개선되지 않아도 최대 10 epoch까지 기다림   그 후에도 개선이 없으면 학습률을 줄이거나 학습을 중단

warmup = 100           # learning rate warm-up 단계 수 (처음엔 천천히 증가)
epoch = 1000           # 학습 시 최대 반복할 epoch 수
clip = 1.0             # gradient clipping 최대 허용 값 (gradient exploding 방지)
weight_decay = 5e-4    # 가중치 감소 계수 (L2 regularization 효과)
inf = float('inf')     # 무한대 값 (early stopping, 최소값 비교 등에 사용)

