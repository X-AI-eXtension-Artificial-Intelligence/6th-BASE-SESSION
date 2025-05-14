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

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx #소스에서 padding 인덱스
        self.trg_pad_idx = trg_pad_idx # 타깃에서 padding 인덱스 
        self.trg_sos_idx = trg_sos_idx # 타깃의 시작 토큰 인덱스 
        self.device = device 
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head, #헤드 수
                               max_len=max_len, #최대 시퀀스 길이 
                               ffn_hidden=ffn_hidden, #feed forward 은닉층 크기 
                               enc_voc_size=enc_voc_size, #인코더 단어집합 크기 
                               drop_prob=drop_prob, #드롭아웃 비율
                               n_layers=n_layers, #인코더 블록 수 
                               device=device) #장치

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head, #헤드수 
                               max_len=max_len,# 최대 시퀀스 길이 
                               ffn_hidden=ffn_hidden, #ffn 은닉층 길이 
                               dec_voc_size=dec_voc_size, #디코더 단어 집합 크기
                               drop_prob=drop_prob, #드롭아웃 비율
                               n_layers=n_layers, #디코더 블로 수 
                               device=device) #연산 장치 

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) #소스 패딩하는 마스크 생성 
        trg_mask = self.make_trg_mask(trg) #타킷 패딩하는 마스크 생성
        enc_src = self.encoder(src, src_mask) #인코더에서 문장 벡터로 변환
        output = self.decoder(trg, enc_src, trg_mask, src_mask) #디코더에서 최종 예측 
        return output #결과 반환 

    def make_src_mask(self, src): #소스 마스크 생성 
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg): # 타깃 마스크 생성 
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1] #타깃 시퀀스 길이 
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask #마스크 두개 합쳐서 최종 내기 
        return trg_mask