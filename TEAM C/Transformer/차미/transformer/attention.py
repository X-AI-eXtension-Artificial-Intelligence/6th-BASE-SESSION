import torch
import torch.nn as nn
import math

'''
Transformer의 포지셔널 인코딩을 -> T5에서 사용하는 것처럼 상대 위치 인코딩으로 변환
-> GPT said 위치 간의 상대적 거리 정보가 더 효과적인 경우가 많음
-> Multi-head Attention에 직접 주입하는 구조

바꾼 방법
1. 각 쿼리-키 쌍의 상대 거리(`i - j`)에 따라 상대 위치 인덱스를 만들고  
2. 그에 대응되는 학습 가능한 bias 벡터를 가져와 attention score에 더함  
   -> forward() 함수 안의 attention score에 relative bias 더해줌
3. 최종 softmax 전에 위치 정보가 반영됨
'''


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

        ## 수정: 상대 위치 bias 임베딩 정의
        self.max_seq_len = max_seq_len
        self.relative_position_bias = nn.Embedding(2 * max_seq_len - 1, h)

    ## 수정: 상대 위치 인덱스 행렬 생성 함수
    def _generate_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        상대 위치 인덱스 행렬 생성
        shape: (seq_len, seq_len)
        값: [-seq_len+1, ..., 0, ..., seq_len-1] → index shift 필요
        """
        range_vec = torch.arange(seq_len)
        rel_pos = range_vec.view(-1, 1) - range_vec.view(1, -1)  # (seq_len, seq_len)
        rel_pos += self.max_seq_len - 1  # 음수를 양수로 변환
        return rel_pos  # (seq_len, seq_len)


    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        seq_len = q.size(1)

        query = self.w_q(q)  # (B, S, D)
        key = self.w_k(k)
        value = self.w_v(v)

        # 멀티 헤드 분리 (B, S, D) → (B, H, S, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # 기본 어텐션 스코어 계산
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, S, S)

        ## 수정: 상대 위치 바이어스 계산 및 스코어에 더하기
        relative_position_matrix = self._generate_relative_positions(seq_len).to(q.device)  # (S, S)
        rel_bias = self.relative_position_bias(relative_position_matrix)  # (S, S, H)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, S, S)
        scores += rel_bias  # 위치 바이어스 더하기

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, value)  # (B, H, S, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.h * self.d_k)

        return self.w_o(x)