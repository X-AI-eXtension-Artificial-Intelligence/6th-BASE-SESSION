"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    
    # 트랜스포머 전체 디코더 모듈 정의 클래스
    # 임베딩 → 디코더 레이어 스택 → 출력 선형 변환까지 수행
    

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        
        # param dec_voc_size: 디코더용 단어 집합 크기
        # param max_len: 최대 시퀀스 길이
        # param d_model: 임베딩 및 모델 차원
        # param ffn_hidden: FFN 내부 은닉 차원
        # param n_head: 멀티 헤드 수
        # param n_layers: 디코더 레이어 수
        # param drop_prob: 드롭아웃 확률
        # param device: 연산 디바이스 (cuda or cpu)
        
        super().__init__()

        # 임베딩: 토큰 임베딩 + 포지셔널 인코딩 + 드롭아웃
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # 디코더 레이어 여러 개를 쌓기 (ModuleList 사용)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

        # 출력층: 각 위치별 벡터를 vocabulary 차원으로 투사
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # param trg: 디코더 입력 (target token indices) [batch_size, trg_len]
        # param enc_src: 인코더 출력 [batch_size, src_len, d_model]
        # param trg_mask: 디코더 마스크 (future masking + padding masking)
        # param src_mask: 소스 마스크 (패딩 마스크 등)
        # return: vocab 차원의 로짓 [batch_size, trg_len, dec_voc_size]
        

        # 디코더 입력 임베딩 처리
        trg = self.emb(trg)  # [batch_size, trg_len, d_model]

        # 디코더 레이어 스택 순차 적용
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 마지막 출력 벡터를 vocab 차원으로 변환
        output = self.linear(trg)

        return output
