"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
'''
구성 요소: 
- 임베딩 (Embedding): 출력 문장을 벡터로 변환
- DecoderLayer 여러 개 (n_layers 개): Self-Attention, Encoder-Decoder Attention 포함
- Linear layer: 디코더 출력 → 단어 예측 (vocab 크기로 변환)
- forward: 임베딩 → 각 디코더 레이어 → linear layer → 최종 출력

'''
'''
[출력 문장 (단어 index)]
   └→ TransformerEmbedding (단어 → 벡터 + 위치 정보)
       └→ DecoderLayer 1
           └→ DecoderLayer 2
               └→ ...
                   └→ DecoderLayer N
                       └→ Linear layer (벡터 → 단어 확률)
                           └→ 최종 출력
'''
import torch
from torch import nn

# DecoderLayer와 TransformerEmbedding import
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    """
    Transformer Decoder 클래스
    - 임베딩 + 여러 DecoderLayer + 출력 Linear layer로 구성
    """
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        Decoder 클래스 초기화 함수
        - 임베딩 레이어, 여러 개의 DecoderLayer, 출력 Linear layer 생성

        :param dec_voc_size: 출력 어휘 사전 크기
        :param max_len: 문장의 최대 길이
        :param d_model: 임베딩 벡터 차원
        :param ffn_hidden: FeedForward 은닉층 차원
        :param n_head: multi-head attention 헤드 수
        :param n_layers: DecoderLayer 개수
        :param drop_prob: dropout 비율
        :param device: 연산 장치 (cpu or cuda)
        """
        super().__init__()

        # 임베딩 레이어 생성
        # 출력 토큰 → 벡터로 변환 + positional encoding 포함
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        # DecoderLayer 여러 개 생성
        # 각 layer: self-attention + encoder-decoder attention + feedforward 포함
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # 출력 Linear layer
        # 디코더 출력 → vocab 크기의 벡터로 변환 (단어 확률 분포 예측)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Decoder의 forward 함수
        - 출력 문장을 임베딩 → 각 DecoderLayer 순차적으로 통과
        - 최종 출력 linear layer 통과

        :param trg: 출력 문장 (batch_size x trg_len)
        :param enc_src: Encoder 출력 (batch_size x src_len x d_model)
        :param trg_mask: 출력 마스크 (batch_size x 1 x trg_len x trg_len)
        :param src_mask: 입력 마스크 (batch_size x 1 x 1 x src_len)
        :return: 디코더 출력 (batch_size x trg_len x dec_voc_size)
        """

        # 출력 문장을 임베딩
        # trg: [batch_size, trg_len] → [batch_size, trg_len, d_model]
        trg = self.emb(trg)

        # 각 DecoderLayer에 trg, enc_src, trg_mask, src_mask 전달
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 최종 출력 linear layer 통과
        # trg: [batch_size, trg_len, d_model] → [batch_size, trg_len, dec_voc_size]
        # pass to LM head
        output = self.linear(trg)

        return output