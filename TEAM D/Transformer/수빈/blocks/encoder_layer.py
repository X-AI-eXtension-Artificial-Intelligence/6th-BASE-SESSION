"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
'''
입력 (x)
 └→ Multi-Head Attention
     └→ Dropout
         └→ Add & Norm
             └→ FeedForward
                 └→ Dropout
                     └→ Add & Norm
                         └→ 출력
'''
from torch import nn

# 필요한 레이어 import
from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer EncoderLayer 클래스
    - Multi-Head Attention + Feed Forward + LayerNorm + Dropout 포함
    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
        EncoderLayer 클래스 초기화 함수

        :param d_model: 임베딩 차원
        :param ffn_hidden: FeedForward 은닉층 차원
        :param n_head: Multi-Head Attention의 head 수
        :param drop_prob: dropout 확률
        """
        super(EncoderLayer, self).__init__()

        # Multi-Head Attention 모듈
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)

        # Attention 뒤 LayerNorm + Dropout
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Position-wise Feed Forward 모듈
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        
        # FeedForward 뒤 LayerNorm + Dropout
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        """
        EncoderLayer의 forward 함수

        :param x: 입력 벡터 (batch_size x seq_len x d_model)
        :param src_mask: 입력 마스크 (batch_size x 1 x 1 x seq_len)
        :return: 출력 벡터 (batch_size x seq_len x d_model)
        """

        # 1. compute self attention
        _x = x      # residual connection을 위해 입력 저장 
        x = self.attention(q=x, k=x, v=x, mask=src_mask) # Query=Key=Value=self
        # shape: (batch_size, seq_len, d_model)
        
        # 2. Dropout → Residual Connection → LayerNorm
        x = self.dropout1(x)   # Dropout
        x = self.norm1(x + _x) # Add & Norm
        
        # 3. positionwise feed forward network
        _x = x           # residual connection을 위해 현재 x 저장
        x = self.ffn(x)  # FeedForward 연산

        # 4. Dropout → Residual Connection → LayerNorm
        x = self.dropout2(x)   # Dropout
        x = self.norm2(x + _x) # Add & Norm

        # 5. EncoderLayer 출력 반환
        return x
