import torch
import torch.nn as nn
import math

# 입력 토큰을 벡터로 임베딩하는 클래스
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model # 임베딩 차원을 의미
        self.embedding = nn.Embedding(vocab_size, d_model) # 임베딩 테이블 정의

    def forward(self, x):
        # 입력 토큰 ID에 대한 임베딩 벡터 반환
        # 논문에서 제안된대로 sqrt(d_model)을 곱해 스케일 조정
        return self.embedding(x) * math.sqrt(self.d_model)

# 위치 정보를 임베딩하는 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # (seq_len, d_model) 크기의 포지셔널 인코딩 행렬 생성
        pe = torch.zeros(seq_len, d_model)
        # 각 위치에 대한 인덱스 벡터 생성 (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # 주기 함수의 주기를 조절하는 div_term 계산 (d_model / 2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 짝수 인덱스에는 사인 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스에는 코사인 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        # 배치 차원 추가 (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        # 학습되지 않는 버퍼로 등록
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 텐서에 위치 인코딩을 더한 후 dropout 적용
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)
