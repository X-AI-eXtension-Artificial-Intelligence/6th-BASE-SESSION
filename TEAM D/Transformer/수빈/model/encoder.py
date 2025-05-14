"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
'''
[입력 문장 (단어 index)]
   └→ TransformerEmbedding (단어 → 벡터 + 위치 정보)
       └→ EncoderLayer 1
           └→ EncoderLayer 2
               └→ ...
                   └→ EncoderLayer N
                       └→ 최종 출력
'''
from torch import nn

# EncoderLayer와 TransformerEmbedding 클래스 import
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    """
    Transformer Encoder 클래스
    - 임베딩 + 여러 EncoderLayer로 구성
    """
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        """
        Encoder 클래스 초기화 함수
        - 입력 문장을 임베딩하고, 여러 EncoderLayer를 쌓아 올림

        :param enc_voc_size: 입력 어휘 사전 크기
        :param max_len: 입력 문장의 최대 길이
        :param d_model: 임베딩 벡터 차원
        :param ffn_hidden: FeedForward 은닉층 차원
        :param n_head: multi-head attention 헤드 수
        :param n_layers: EncoderLayer 개수
        :param drop_prob: dropout 비율
        :param device: 연산 장치 (cpu or cuda)
        """
        
        super().__init__()
        # 임베딩 레이어 생성
        # 입력 토큰 → 벡터로 변환 + positional encoding 포함
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # EncoderLayer 여러 개를 리스트로 생성
        # 각 EncoderLayer는 attention + feedforward + layer norm 포함
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])   # n_layers 개수만큼 쌓기

    def forward(self, x, src_mask):
        """
        Encoder의 forward 함수
        - 입력 문장을 임베딩 → 각 EncoderLayer 순차적으로 통과

        :param x: 입력 문장 (batch_size x seq_len)
        :param src_mask: 입력 마스크 (batch_size x 1 x 1 x seq_len)
        :return: 인코딩된 입력 (batch_size x seq_len x d_model)
        """

        # 입력 문장 임베딩
        # x: [batch_size, seq_len] → [batch_size, seq_len, d_model]
        x = self.emb(x)

        # 각 EncoderLayer에 x와 src_mask 전달
        # EncoderLayer 내부: multi-head attention → add & norm → feedforward → add & norm
        for layer in self.layers:
            x = layer(x, src_mask)

        # 최종 인코더 출력 반환
        return x
    
