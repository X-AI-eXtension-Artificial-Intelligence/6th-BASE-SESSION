"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,       # 차원:d_model
            max_len=max_len,       # 최대길이:max_len
            vocab_size=enc_voc_size,  # 어휘크기:enc_voc_size
            drop_prob=drop_prob,    # 드롭확률:drop_prob
            device=device          # 디바이스:device
        )

        self.layers = nn.ModuleList([  # 모듈리스트:레이어들
            EncoderLayer(
                d_model=d_model,       # 차원:d_model
                ffn_hidden=ffn_hidden, # FFN은닉:ffn_hidden
                n_head=n_head,         # 헤드수:n_head
                drop_prob=drop_prob    # 드롭확률:drop_prob
            ) for _ in range(n_layers)  # 반복:레이어수
        ])


    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x