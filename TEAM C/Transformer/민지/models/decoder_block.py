import torch.nn as nn

from .layers.layer_norm import LayerNorm
from .layers.multi_head_attention import MultiHeadAttention
from .layers.feed_forward import FeedForward


class DecoderBlock(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, drop_prob=drop_prob)
        self.norm1 = LayerNorm(embedding_dim=d_model)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(embedding_dim=d_model)

        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(embedding_dim=d_model)

    def forward(self, dec, enc, trg_mask, src_mask):
        # masked self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # add and norm
        x = self.norm1(x=x, sublayer_output=_x)

        # cross attention
        if enc is not None:
            # encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.norm2(x=x, sublayer_output=_x)

        # position-wise ffn
        _x = x
        x = self.ffn(x)
        x = self.norm3(x=x, sublayer_output=_x)

        return x