import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module): # Transformer 모델 클래스 정의

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx # 소스 패딩 토큰 인덱스 저장 
        self.trg_pad_idx = trg_pad_idx # 타켓 패딩 토큰 인덱스 저장
        self.trg_sos_idx = trg_sos_idx # 타겟 시작 토큰 인덱스 저장
        self.device = device
        # Encoder 객체 생성, 저장
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # Decoder 객체 생성, 저장
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) # 소스 시퀀스의 마스킹 텐서 저장
        trg_mask = self.make_trg_mask(trg) # 타겟 시퀀스의 마스킹 텐서 저장
        enc_src = self.encoder(src, src_mask) # 인코더에 입력 시퀀스와 마스크 전달 -> 인코더 출력 get
        output = self.decoder(trg, enc_src, trg_mask, src_mask) # 디코더에 타겟 시퀀스, 인코더 출력, 마스크들을 전달 -> 최종 출력 get
        return output # 모델의 최종 출력 반환

    def make_src_mask(self, src): # 소스 마스크 생성 함수 정의
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # 패딩 토큰이 아닌 위치는 True, 패딩 토큰은 False로 하는 마스크 생성
        return src_mask # 소스 마스크 반환

    def make_trg_mask(self, trg): # 타겟 마스크 생성 함수 정의
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1] # 타겟 시퀀스 길이
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device) # 자기회귀 마스킹을 위한 하삼각 행렬 생성
        trg_mask = trg_pad_mask & trg_sub_mask # 패딩 마스크 + 자기회귀 마스크 결합
        return trg_mask # 최종 타겟 마스크 반환