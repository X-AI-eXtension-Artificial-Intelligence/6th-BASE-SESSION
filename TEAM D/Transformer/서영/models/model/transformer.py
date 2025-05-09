"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

# 인코더와 디코더 모듈 import
from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):
    # 전체 Transformer 모델 정의 (Encoder-Decoder 구조 기반)

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        # 패딩 토큰 인덱스 (마스킹에 사용)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # 디코더 입력 시작 토큰 (sos: start of sentence)
        self.trg_sos_idx = trg_sos_idx

        # 연산 디바이스
        self.device = device

        # 인코더 정의
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

        # 디코더 정의
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
        # src: 인코더 입력 (소스 시퀀스)
        # trg: 디코더 입력 (타겟 시퀀스)

        src_mask = self.make_src_mask(src)  # 패딩 토큰 마스킹 (인코더용)
        trg_mask = self.make_trg_mask(trg)  # 패딩 + 미래 단어 마스킹 (디코더용)

        enc_src = self.encoder(src, src_mask)  # 인코더 출력 (context vector)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)  # 디코더 예측 결과

        return output  # shape: [batch_size, trg_seq_len, vocab_size]

    def make_src_mask(self, src):
        # src: [batch_size, src_seq_len]
        # 패딩 위치를 False로 마스킹
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # shape: [batch_size, 1, 1, src_seq_len] → broadcast용
        return src_mask

    def make_trg_mask(self, trg):
        # trg: [batch_size, trg_seq_len]

        # 패딩 토큰 마스킹
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # shape: [batch_size, 1, trg_seq_len, 1]

        trg_len = trg.shape[1]

        # 미래 정보 보지 않도록 아래 삼각행렬 마스킹 생성
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        # shape: [trg_seq_len, trg_seq_len]

        # 두 마스크를 AND 연산 → padding + look-ahead mask 동시 적용
        trg_mask = trg_pad_mask & trg_sub_mask
        # shape: [batch_size, 1, trg_seq_len, trg_seq_len]
        return trg_mask
