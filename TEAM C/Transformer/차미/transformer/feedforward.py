import torch.nn as nn
import torch

# 포지션별 피드포워드 네트워크 정의
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # 첫 번째 선형 레이어
        self.dropout = nn.Dropout(dropout) # 드롭아웃 적용
        self.linear_2 = nn.Linear(d_ff, d_model) # 두 번째 선형 레이어

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        # return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        ## relu에서 gelu로 수정
        return self.linear_2(self.dropout(torch.gelu(self.linear_1(x))))
