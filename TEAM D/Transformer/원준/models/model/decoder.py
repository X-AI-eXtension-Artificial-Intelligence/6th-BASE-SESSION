
import torch
from torch import nn 

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    Transformer 디코더 전체를 구현한 클래스
    - 입력: 디코더 입력 시퀀스 (trg), 인코더 출력 (enc_src)
    - 출력: 각 위치별 다음 단어 예측을 위한 로짓 (logit)
    """

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 1. 입력 시퀀스에 대해 임베딩 + 위치 인코딩 적용
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # 2. 여러 개의 DecoderLayer 쌓기
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

        # 3. 디코더 마지막 출력 → vocab 크기로 선형 변환 (언어 모델 헤드)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        :param trg: 디코더 입력 시퀀스 (target token indices)        [batch, trg_len]
        :param enc_src: 인코더 출력 (context)                        [batch, src_len, d_model]
        :param trg_mask: 디코더용 마스크 (look-ahead + padding)     [batch, 1, trg_len, trg_len]
        :param src_mask: 인코더 입력에 대한 마스크 (padding)        [batch, 1, 1, src_len]
        :return: 디코더 최종 로짓 출력 (단어 예측을 위한 결과)      [batch, trg_len, dec_voc_size]
        """

        # 1. 입력 임베딩 + 포지셔널 인코딩
        trg = self.emb(trg)  # shape: [batch, trg_len, d_model]

        # 2. 모든 DecoderLayer를 순차적으로 통과
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 3. 출력 벡터를 vocab 차원으로 선형 변환 (로짓 생성)
        output = self.linear(trg)  # shape: [batch, trg_len, dec_voc_size]

        return output  # 최종 예측 로짓 반환
