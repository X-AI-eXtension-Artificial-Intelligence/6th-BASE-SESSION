from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        # d_model: 모델 차원 (예: 512)
        # ffn_hidden: FFN에서 내부 은닉층 차원 (예: 2048)
        # n_head: Multi-Head Attention의 head 개수 (예: 8)
        # drop_prob: drop 확률 (예: 0.1)
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        
        # Self-Attention 수행
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # Add + Norm
        x = self.dropout1(x)
        x = self.norm1(x + _x) # Residual Connection
        
        # Feed Forward Network(FFN) 적용
        _x = x
        x = self.ffn(x)
      
        # Add + Norm
        x = self.dropout2(x)
        x = self.norm2(x + _x) # Residual Connection
        return x
