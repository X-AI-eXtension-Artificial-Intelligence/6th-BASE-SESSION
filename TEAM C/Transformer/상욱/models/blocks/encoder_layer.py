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
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.norm2 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # --- Step 1: Self-Attention with Pre-LayerNorm
        norm_x = self.norm1(x)
        attn_out = self.attention(q=norm_x, k=norm_x, v=norm_x, mask=src_mask)
        x = x + self.dropout1(attn_out)  # Residual + Dropout

        # --- Step 2: Positionwise FFN with Pre-LayerNorm
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout2(ffn_out)  # Residual + Dropout

        return x
# Pre-LayerNorm 구조 적용 -> 각 sub-layer 앞에 LayerNorm을 먼저 적용해 학습 안정성 향상
# Residual connection은 유지 -> LayerNorm 전에 입력을 복사하여 잔차 연결에 사용
# Dropout은 sub-layer 후에 적용