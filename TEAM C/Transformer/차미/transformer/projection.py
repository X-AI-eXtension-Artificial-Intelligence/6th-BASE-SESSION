import torch.nn as nn

# 최종 출력 벡터를 단어 분포로 변환하는 선형 레이어
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # 단어 수만큼의 출력을 생성

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return self.proj(x)
