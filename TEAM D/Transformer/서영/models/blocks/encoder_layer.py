"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

# Transformer 구성 요소들을 import
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    # Transformer의 인코더에서 하나의 레이어를 정의한 클래스
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        # 1. 셀프 어텐션 레이어 정의 (입력 x에 대해 q, k, v 모두 동일)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)           # 첫 번째 레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob)           # 첫 번째 드롭아웃

        # 2. 포지션 와이즈 피드포워드 네트워크 정의
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)           # 두 번째 레이어 정규화
        self.dropout2 = nn.Dropout(p=drop_prob)           # 두 번째 드롭아웃

    def forward(self, x, src_mask):
        # 인코더 입력 x와 마스크 src_mask를 받아 처리

        # 1. 셀프 어텐션 수행 (자기 자신에 대한 어텐션)
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. 잔차 연결 후 정규화
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. 포지션 와이즈 피드포워드 네트워크 수행
        _x = x
        x = self.ffn(x)

        # 4. 잔차 연결 후 정규화
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x  # 출력은 다음 인코더 레이어 또는 디코더로 전달
