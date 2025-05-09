"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# 디코더의 한 레이어와 임베딩 모듈 불러오기
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    # Transformer 구조의 디코더 전체 모듈 정의

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 디코더 입력 시퀀스에 대한 임베딩 (토큰 임베딩 + 위치 인코딩)
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device
        )

        # 디코더 레이어를 n_layers 만큼 쌓음
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

        # 최종 출력층: hidden state를 vocab 크기로 projection → 각 토큰에 대한 확률 분포
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: 디코더 입력 시퀀스 (토큰 인덱스)
        # enc_src: 인코더의 최종 출력 (context vector)
        # trg_mask: 디코더 셀프 어텐션 마스크
        # src_mask: 인코더-디코더 어텐션 마스크

        trg = self.emb(trg)  # 입력 시퀀스를 임베딩

        for layer in self.layers:
            # 각 디코더 레이어를 통과하면서 어텐션 및 FFN 수행
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 최종 hidden state를 vocab 크기로 투영하여 토큰 확률 분포 생성
        output = self.linear(trg)
        return output  # shape: [batch_size, seq_len, vocab_size]
