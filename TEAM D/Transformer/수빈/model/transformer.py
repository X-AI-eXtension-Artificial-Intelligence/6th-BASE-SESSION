"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

'''
입력 문장 (src)
└→ make_src_mask (PAD 마스크)
└→ Encoder (src + src_mask → enc_src)

출력 문장 (trg)
└→ make_trg_mask (PAD + future 마스크)
└→ Decoder (trg + enc_src + trg_mask + src_mask → output)

output 반환
'''

import torch
from torch import nn


# Encoder와 Decoder 모듈 import (다른 파일에 정의되어 있음)
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """
    Transformer 전체 모델 클래스
    Encoder + Decoder 구조를 포함
    """
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # Encoder 객체 생성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # Decoder 객체 생성
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        """
        Transformer 모델의 forward 함수
        - 입력 문장을 인코더에 통과
        - 인코더의 출력과 타겟 문장을 디코더에 전달
        - 디코더의 출력 반환

        :param src: 입력 문장 (batch_size x src_len)
        :param trg: 출력 문장 (batch_size x trg_len)
        :return: Transformer 모델의 출력 (batch_size x trg_len x vocab_size)
        """
        # 입력 문장 마스크 생성 (PAD 토큰 부분을 마스킹)
        src_mask = self.make_src_mask(src)
        # 출력 문장 마스크 생성 (PAD 토큰 + future token 마스킹)
        trg_mask = self.make_trg_mask(trg)
        # 인코더에 입력과 입력 마스크 전달 → 인코더 출력 계산
        enc_src = self.encoder(src, src_mask)
        # 디코더에 타겟, 인코더 출력, 마스크 전달 → 최종 출력 계산
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output


    def make_src_mask(self, src):
        """
        입력 문장 마스크 생성 함수
        - PAD 토큰이 아닌 부분은 True (1)
        - PAD 토큰인 부분은 False (0)

        :param src: 입력 문장 (batch_size x src_len)
        :return: src_mask (batch_size x 1 x 1 x src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        출력 문장 마스크 생성 함수
        - future token 마스킹 + PAD 마스킹

        :param trg: 출력 문장 (batch_size x trg_len)
        :return: trg_mask (batch_size x 1 x trg_len x trg_len)
        """
        # PAD 마스크: PAD가 아닌 위치 True
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # 시퀀스 길이
        trg_len = trg.shape[1]

        # future token 마스크 (하삼각 행렬)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()

        # 최종 마스크 = PAD 마스크 & future token 마스크
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
