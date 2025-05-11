"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    
    # 트랜스포머 전체 모델 클래스
    # 인코더 + 디코더 + 마스킹 로직 포함
    

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        
        # param src_pad_idx: 소스 문장의 PAD 토큰 인덱스
        # param trg_pad_idx: 타겟 문장의 PAD 토큰 인덱스
        # param trg_sos_idx: 디코더 입력 시작 토큰(SOS) 인덱스
        # param enc_voc_size: 인코더 단어 집합 크기
        # param dec_voc_size: 디코더 단어 집합 크기
        # param d_model: 임베딩 및 모델 차원
        # param n_head: 멀티 헤드 수
        # param max_len: 최대 시퀀스 길이
        # param ffn_hidden: FFN 내부 차원
        # param n_layers: 인코더/디코더 층 수
        # param drop_prob: 드롭아웃 확률
        # param device: 연산 디바이스 (cpu or cuda)
        
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 인코더 구성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # 디코더 구성
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        
        # 전체 Transformer 모델의 순전파 함수

        # param src: 인코더 입력 (batch_size, src_len)
        # param trg: 디코더 입력 (batch_size, trg_len)
        # return: vocab 차원의 로짓 (batch_size, trg_len, dec_voc_size)
        
        src_mask = self.make_src_mask(src)     # PAD 마스킹
        trg_mask = self.make_trg_mask(trg)     # future 마스킹 + PAD 마스킹

        enc_src = self.encoder(src, src_mask)  # 인코더 실행
        output = self.decoder(trg, enc_src, trg_mask, src_mask)  # 디코더 실행

        return output

    def make_src_mask(self, src):
        
        # 소스 시퀀스에서 PAD 토큰을 가리는 마스크 생성
        # shape: [batch_size, 1, 1, src_len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        
        # PAD 토큰 마스크
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # 미래 정보 차단을 위한 하삼각 행렬
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)

        # 두 마스크를 AND 연산으로 결합
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
