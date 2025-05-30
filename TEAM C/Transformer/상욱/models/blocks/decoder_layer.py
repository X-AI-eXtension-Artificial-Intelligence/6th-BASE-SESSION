"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        # Pre-LN 적용
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)

        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # --- Step 1: Self-Attention (with Pre-LN)
        norm_dec = self.norm1(dec)
        self_attn = self.self_attention(q=norm_dec, k=norm_dec, v=norm_dec, mask=trg_mask)
        x = dec + self.dropout1(self_attn)  # Residual

        # --- Step 2: Encoder-Decoder Attention (with Pre-LN)
        norm_x = self.norm2(x)
        if enc is not None:
            enc_dec_attn = self.enc_dec_attention(q=norm_x, k=enc, v=enc, mask=src_mask)
            x = x + self.dropout2(enc_dec_attn)  # Residual

        # --- Step 3: Fusion (Self + Enc-Dec Attention already merged into x above)
        #           → Apply FFN to fused result with Pre-LN
        norm_x = self.norm3(x)
        x = x + self.dropout3(self.ffn(norm_x))  # Final Residual

        return x
    
# Pre-LN -> 학습 안정성
# Attention 통합 구조 -> 표현력 및 정보 융합 능력 강화