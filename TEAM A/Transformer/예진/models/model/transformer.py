"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

# 인코더 디코더 구조 불러오기
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    """
    전체 Transformer 모델 클래스
    - Encoder와 Decoder를 연결하여 전체 입력-출력 변환 수행
    - 문장 인코딩, 마스크 생성, 문장 생성
    """

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx,
                 enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        """
        src_pad_idx: 입력 문장에서 padding 토큰의 인덱스
        trg_pad_idx: 출력 문장에서 padding 토큰의 인덱스
        trg_sos_idx: 디코더 입력 시작 토큰의 인덱스
        enc_voc_size: 인코더에서 사용할 단어 사전 크기
        dec_voc_size: 디코더에서 사용할 단어 사전 크기
        d_model: 벡터 차원 크기
        n_head: attention 헤드 수
        max_len: 문장 최대 길이
        ffn_hidden: Feed Forward Network 은닉층 크기
        n_layers: 인코더/디코더 층 개수
        drop_prob: dropout 비율
        device: CPU 또는 GPU
        """
        super().__init__()

        # 입력 시퀀스와 출력 시퀀스의 padding 토큰 인덱스 저장
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # 디코더 시작 토큰 인덱스 저장 (추론 시 활용)
        self.trg_sos_idx = trg_sos_idx

        # 장치 설정
        self.device = device

        # 인코더 생성: 입력 문장을 context vector로 변환
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

        # 디코더 생성: context vector와 이전 단어들을 바탕으로 다음 단어 예측
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device
        )

    def forward(self, src, trg):
        """
        Transformer 순전파 함수

        src: 입력 문장 (예: 영어)
        trg: 출력 문장 (예: 프랑스어), 이전 시점까지의 디코더 입력

        - 입력: src -> encoder -> 인코더 출력 enc_src
        - 출력: trg + enc_src -> decoder -> 다음 단어 확률 분포
        """

        # 입력 문장에 대한 마스크 생성 (패딩 위치 무시)
        src_mask = self.make_src_mask(src)

        # 출력 문장에 대한 마스크 생성 (미래 단어 가리기 + 패딩 가리기)
        trg_mask = self.make_trg_mask(trg)

        # 인코더 통과 (문장 내 단어 관계를 반영한 벡터 생성)
        enc_src = self.encoder(src, src_mask)

        # 디코더 통과 (이전 단어들과 인코더 출력 바탕으로 다음 단어 예측)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output

    def make_src_mask(self, src):
        """
        입력 마스크 생성 함수
        - src에 있는 padding 토큰의 위치를 0으로 표시
        - 이후 attention 연산에서 해당 위치는 무시
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        출력 마스크 생성 함수
          (1) padding 토큰 가리기
          (2) 디코더가 미래 단어를 보지 못하도록 하삼각 마스크 적용
        """
        # 1. padding 마스크: padding 위치는 False로 설정
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        # 2. 하삼각 마스크: 자기 자신보다 이후 위치는 가림
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)

        # 3. 두 마스크 결합
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
