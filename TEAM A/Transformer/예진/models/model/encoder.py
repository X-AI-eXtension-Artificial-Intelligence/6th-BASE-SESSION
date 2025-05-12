"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

from torch import nn

# 인코더 층 (Self-Attn + FFN) 정의한 모듈 불러오기
from models.blocks.encoder_layer import EncoderLayer

from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer의 전체 인코더 구성하는 클래스

    입력 문장을 여러 층의 EncoderLayer에 통과시켜서
    문장 안의 문맥 정보를 반영한 벡터로 변환
    """

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        enc_voc_size: 입력 단어 개수
        max_len: 문장 최대 길이 (포지셔널 인코딩용)
        d_model: 임베딩 및 attention 연산에서 사용하는 차원 크기
        ffn_hidden: Feed Forward Network 은닉층 크기
        n_head: Multi-Head Attention에서 나눌 헤드 수
        n_layers: EncoderLayer 몇 층 쌓을지
        drop_prob: dropout 비율
        device: 연산 장치 (GPU 또는 CPU)
        """
        super().__init__()

        # 1. 입력 임베딩 + 위치 인코딩
        # 입력된 단어 ID를 벡터로 변환 + 위치 정보 추가
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device
        )

        # 2. EncoderLayer를 여러 개 쌓음
        # 각 층은 self-attention → feed forward 구조
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        """
        x: 입력 문장의 단어 ID 시퀀스
        src_mask: padding 위치 등을 가리기 위한 마스크
        """

        # 1. 단어 임베딩 + 위치 인코딩 적용
        x = self.emb(x)

        # 2. 각 EncoderLayer를 순서대로 통과시킴
        # 각 층마다 문장 내부 단어들 간의 문맥 관계를 점점 더 깊게 학습
        for layer in self.layers:
            x = layer(x, src_mask)

        # 최종 인코더 출력 (문맥 반영된 상태)
        return x
