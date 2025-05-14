"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

# 인코더 레이어와 임베딩 모듈 불러오기
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    # Transformer 구조의 인코더 전체 모듈 정의

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 입력 시퀀스에 대한 임베딩 정의 (토큰 임베딩 + 위치 인코딩)
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device
        )

        # 인코더 레이어 n_layers 개 쌓기 (Multi-Head Attention + FFN + LayerNorm)
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        # x: 입력 시퀀스 (토큰 인덱스)
        # src_mask: 패딩 등 마스킹할 위치 지정

        x = self.emb(x)  # 임베딩: [batch_size, seq_len, d_model]

        for layer in self.layers:
            # 각 인코더 레이어를 통과하며 어텐션 및 FFN 수행
            x = layer(x, src_mask)

        return x  # 인코더의 최종 출력 (context vector)
