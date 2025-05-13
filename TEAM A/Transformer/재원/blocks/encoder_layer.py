from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


# 인코더 레이어(블록)의 전반적인 설계

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob): # 초기 파라미터는 입력 차원, 헤드 수, 드롭아웃 비율, FFN
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head) #d_model 차원의 인풋 받아서, 지정된 헤드 수 고려하여 Multi-head Attention 수행
        self.norm1 = LayerNorm(d_model=d_model) #레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob) #드롭아웃

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # Feed forward  
        self.norm2 = LayerNorm(d_model=d_model) # 레이어 정규화
        self.dropout2 = nn.Dropout(p=drop_prob) # 드롭아웃

    # 인코더 블록 하나 순전파 설계
    def forward(self, x, src_mask):
        # 1. Self attetion 계산
        _x = x # Residual Connection을 위한 초기 입력값 복사
        x = self.attention(q=x, k=x, v=x, mask=src_mask) # 인코더에서의 마스킹 -> <PAD> 토큰 무시하도록 지정(가변 길이 시퀀스)
        
        # 2. Add and Norm
        x = self.dropout1(x) # 드롭아웃하고
        x = self.norm1(x + _x) # Residual Connection 해서 나온 결과값에 레이어 정규화
        
        # 3. positionwise feed forward network
        _x = x # 다시 해당 시점의 값 복사
        x = self.ffn(x) # FFN
      
        # 4. Add and Norm
        x = self.dropout2(x) # 드롭아웃하고고
        x = self.norm2(x + _x) # Residual Connection 해서 나온 결과값에 레이어 정규화
        return x
