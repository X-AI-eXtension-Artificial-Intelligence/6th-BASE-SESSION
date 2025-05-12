"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

# 디코더의 층 (Self-Attn, Enc-Dec-Attn, FFN 구성) 불러오기
from models.blocks.decoder_layer import DecoderLayer

from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    전체 Transformer 디코더 모듈 정의

    역할:
    - 입력된 토큰(예: 이전까지 생성된 단어들) 임베딩
    - 여러 개의 DecoderLayer 통과시켜 문맥을 반영한 벡터로 변환
    - 최종적으로 단어 집합 크기만큼의 분포로 바꿈 (단어 예측용)
    """

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        dec_voc_size: 디코더가 사용할 단어 크기
        max_len: 위치 인코딩에서 사용할 최대 문장 길이
        d_model: 임베딩 및 attention에서 사용할 벡터 차원 크기
        ffn_hidden: FFN 내부 은닉층 차원 크기
        n_head: multi-head attention의 헤드 수
        n_layers: DecoderLayer 몇 층 쌓을지
        drop_prob: dropout 확률
        device: 연산 수행할 장치 (CPU 또는 GPU)
        """
        super().__init__()

        # 1. 입력 임베딩 + 포지셔널 인코딩
        # 입력된 단어 ID를 d_model 차원의 벡터로 변환 + 위치 정보 추가
        self.emb = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device
        )

        # 2. DecoderLayer 여러 층 쌓기
        # DecoderLayer를 n_layers만큼 리스트로 저장, 각 층은 self-attn, enc-dec-attn, ffn 구조
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

        # 3. 출력 선형변환
        # 각 위치의 벡터를 단어 분포로 변환 (softmax 이전 단계)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        trg: 디코더 입력 시퀀스 (예: 이전에 생성된 단어들), shape: [batch, trg_len]
        enc_src: 인코더 출력, shape: [batch, src_len, d_model]
        trg_mask: 디코더 내부 마스크 (미래 단어 가리기)
        src_mask: 인코더 출력 마스크 (padding 위치 가리기)
        """

        # 1. 단어 ID에 임베딩 + 위치 인코딩 적용
        # shape: [batch, trg_len, d_model]
        trg = self.emb(trg)

        # 2. DecoderLayer들을 순차적으로 통과
        # 각 층은 자기 자신에 대한 attention, 인코더와의 attention, feed forward 포함
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 3. 각 위치의 벡터를 단어 분포로 변환
        # shape: [batch, trg_len, vocab_size]
        output = self.linear(trg)

        return output  # softmax는 밖에서 처리
