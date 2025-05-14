import torch.nn as nn
from .layers.layer_norm import LayerNorm
from .layers.multi_head_attention import MultiHeadAttention
from .layers.feed_forward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob)
        self.norm1 = LayerNorm(embedding_dim=d_model)

        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(embedding_dim=d_model)

    def forward(self, x, src_mask):
        # self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # add and norm
        x = self.norm1(x=x, sublayer_output=_x)

        # position-wise ffn
        _x = x
        x = self.ffn(x)

        # add and norm
        x = self.norm2(x=x, sublayer_output=_x)

        return x