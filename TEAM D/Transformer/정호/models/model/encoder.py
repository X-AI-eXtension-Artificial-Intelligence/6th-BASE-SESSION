"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    
    # 트랜스포머 인코더 전체 모듈
    # 입력 토큰 → 임베딩 → 여러 인코더 레이어 통과 → 최종 인코더 출력 생성
    

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        
        # param enc_voc_size: 인코더용 단어 집합 크기 (vocab size)
        # param max_len: 최대 시퀀스 길이
        # param d_model: 모델의 차원 수 (임베딩 차원 포함)
        # param ffn_hidden: FFN 내부 은닉 차원
        # param n_head: 멀티 헤드 어텐션의 헤드 수
        # param n_layers: 인코더 레이어 개수
        # param drop_prob: 드롭아웃 비율
        # param device: 실행 디바이스 (cpu 또는 cuda)
        
        super().__init__()

        # 임베딩: 단어 임베딩 + 포지셔널 인코딩 포함
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 인코더 레이어들을 리스트로 구성 (n_layers 만큼 쌓음)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        
        # param x: 입력 시퀀스 토큰 인덱스 (batch_size, src_len)
        # param src_mask: 패딩 토큰을 가리기 위한 마스크 (batch_size, 1, 1, src_len)
        # return: 인코더 출력 (batch_size, src_len, d_model)
        
        # 임베딩 적용 (토큰 + 포지셔널 + 드롭아웃)
        x = self.emb(x)  # shape: (batch_size, src_len, d_model)

        # 각 인코더 레이어를 통과시키며 정보를 정제
        for layer in self.layers:
            x = layer(x, src_mask)

        # 마지막 인코더 출력 반환
        return x
