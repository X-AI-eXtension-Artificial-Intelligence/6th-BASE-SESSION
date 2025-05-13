"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):  # 클래스:Transformer

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):  # 초기화
        super().__init__()
        self.src_pad_idx = src_pad_idx  # 패딩인덱스:소스
        self.trg_pad_idx = trg_pad_idx  # 패딩인덱스:타겟
        self.trg_sos_idx = trg_sos_idx  # 시작토큰인덱스:타겟
        self.device = device  # 디바이스
        self.encoder = Encoder(d_model=d_model,  # 인스턴스:Encoder
                               n_head=n_head,  # 헤드수
                               max_len=max_len,  # 최대길이
                               ffn_hidden=ffn_hidden,  # FFN은닉
                               enc_voc_size=enc_voc_size,  # 어휘크기:인코더
                               drop_prob=drop_prob,  # 드롭확률
                               n_layers=n_layers,  # 레이어수
                               device=device)  # 디바이스

        self.decoder = Decoder(d_model=d_model,  # 인스턴스:Decoder
                               n_head=n_head,  # 헤드수
                               max_len=max_len,  # 최대길이
                               ffn_hidden=ffn_hidden,  # FFN은닉
                               dec_voc_size=dec_voc_size,  # 어휘크기:디코더
                               drop_prob=drop_prob,  # 드롭확률
                               n_layers=n_layers,  # 레이어수
                               device=device)  # 디바이스

    def forward(self, src, trg):  # 순전파
        src_mask = self.make_src_mask(src)  # 소스마스크생성
        trg_mask = self.make_trg_mask(trg)  # 타겟마스크생성
        enc_src = self.encoder(src, src_mask)  # 인코더출력
        output = self.decoder(trg, enc_src, trg_mask, src_mask)  # 디코더출력
        return output  # 반환:출력

    def make_src_mask(self, src):  # 함수:소스마스크
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # 패딩마스크생성
        return src_mask  # 반환:소스마스크

    def make_trg_mask(self, trg):  # 함수:타겟마스크
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)  # 패딩마스크
        trg_len = trg.shape[1]  # 길이:타겟
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)  # 서브마스크:LowerTriangle
        trg_mask = trg_pad_mask & trg_sub_mask  # 결합:패딩마스크 & 서브마스크
        return trg_mask  # 반환:타겟마스크
