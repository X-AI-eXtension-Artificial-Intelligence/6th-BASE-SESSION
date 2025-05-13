import torch.nn as nn

from .encoder_block import EncoderBlock
from .embedding.input_embedding import InputEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = InputEmbedding(d_model=d_model,
                                  max_len=max_len,
                                  vocab_size=enc_voc_size,
                                  drop_prob=drop_prob,
                                  device=device)

        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x