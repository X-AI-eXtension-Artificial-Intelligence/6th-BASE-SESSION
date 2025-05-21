import torch
import torch.nn as nn
import math

# 멀티 헤드 어텐션 수행하는 클래스
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model이 헤드 수로 나누어 떨어져야 함"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h # 각 헤드에서 사용할 차원 수

        self.w_q = nn.Linear(d_model, d_model, bias=False) # 쿼리 생성
        self.w_k = nn.Linear(d_model, d_model, bias=False) # 키 생성
        self.w_v = nn.Linear(d_model, d_model, bias=False) # 밸류 생성
        self.w_o = nn.Linear(d_model, d_model, bias=False) # 출력 가중치

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # 점수 행렬 계산
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # 마스킹 처리
        scores = scores.softmax(dim=-1) # 확률화
        if dropout is not None:
            scores = dropout(scores)
        return scores @ value, scores

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.w_o(x)
