import torch.nn as nn
from .normalization import LayerNormalization

# 잔차 연결과 정규화를 가티 수행
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization(features) # 정규화
        self.dropout = nn.Dropout(dropout) # 드롭아웃으로 과적합 방지

    def forward(self, x, sublayer):
        # 입력을 정규화한 후 서브레이어를 거쳐 드롭아웃을 적용하고 잔차 연결 수행
        return x + self.dropout(sublayer(self.norm(x)))
