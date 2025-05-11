"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    
    # 트랜스포머 인코더의 한 층을 정의하는 클래스
    # 각 인코더 레이어는 셀프 어텐션과 포지션 와이즈 피드포워드 네트워크로 구성됨됨
    

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        # 셀프 어텐션 레이어
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)             # 첫 번째 레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob)             # 드롭아웃으로 과적합 방지

        # 포지션 와이즈 피드포워드 네트워크
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)             # 두 번째 레이어 정규화
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask):
        
        # x: 입력 텐서 (batch_size, src_len, d_model)
        # mask: 마스킹 텐서 (패딩 등 무시할 위치 표시)
        

        # 셀프 어텐션 수행 (인코더 내부에서 토큰 간 상호작용)
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=mask)

        # 잔차 연결(residual connection) + 정규화
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 포지션 와이즈 피드포워드 네트워크
        _x = x
        x = self.ffn(x)

        # 잔차 연결 + 정규화
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x
