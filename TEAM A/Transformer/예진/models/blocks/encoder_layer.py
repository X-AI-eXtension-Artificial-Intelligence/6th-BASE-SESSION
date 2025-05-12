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
    """
    Transformer의 인코더 층
    구성: 
    Self-Attention -> Add & Norm -> Feed Forward -> Add & Norm
    1) Self-Attention: 문장 안의 단어들끼리 서로 영향을 주도록 함
    2) Feed Forward Network: 각 단어별로 독립적인 의미 변형 수행
    각 부분마다 Add & Norm 구조가 붙음 (잔차 연결 + 정규화)
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        d_model: 입력 임베딩 차원 
        ffn_hidden: Feed Forward 내부 은닉층 크기
        n_head: attention에서 몇 개의 헤드로 나눠서 볼지
        drop_prob: dropout 확률 (과적합 방지용)
        """
        super(EncoderLayer, self).__init__()

        # 1. Multi-Head Self-Attention Layer: 하나의 문장 안에서 단어들끼리 서로 정보를 주고받도록 함
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)         # 첫 번째 Layer Normalization(학습 안정성 향상)
        self.dropout1 = nn.Dropout(p=drop_prob)         # 첫 번째 Dropout (과적합 방지)

        # 2. Position-wise Feed Forward Network
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)         # 두 번째 Layer Normalization
        self.dropout2 = nn.Dropout(p=drop_prob)         # 두 번째 Dropout

    def forward(self, x, src_mask):
        """
        forward 함수는 입력을 받아 EncoderLayer 하나를 통과시킴

        x: 입력 문장 벡터들 (형태: [batch_size, 문장길이, d_model])
        src_mask: 입력 문장에서 padding 위치를 가려주는 마스크 (주의할 필요 없는 위치는 제외시킴)
        """

        # 1. self attention 계산 
        # 입력 벡터 x를 Query, Key, Value로 넣어서 attention 계산
        _x = x  # 나중에 잔차 연결 위해 원본을 따로 저장해둠
        x = self.attention(q=x, k=x, v=x, mask=src_mask)  # src_mask로패딩 위치 무시

        # 2. 잔차 연결 후 정규화
        # attention 결과에 dropout 적용하고, 원래 입력값과 더한 후 정규화
        x = self.dropout1(x)         # dropout: 모델 일반화
        x = self.norm1(x + _x)       # 원래 입력값과 더해줌 -> 잔차 연결

        # 3. Feed Forward Network
        # attention 결과를 각 단어마다 독립적인 작은 신경망에 통과시켜 표현 강화
        _x = x                      # 잔차 연결을 위해 저장
        x = self.ffn(x)             # 비선형 변환

        # 4. 잔차 연결 후 정규화
        x = self.dropout2(x)         # dropout
        x = self.norm2(x + _x)       # 두 번째 잔차 연결 + 정규화

        return x  
