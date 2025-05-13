"""
Encoder + Decoder를 모두 포함하는 전체 모델
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx  # 소스 패딩 토큰 인덱스
        self.trg_pad_idx = trg_pad_idx  # 타겟 패딩 토큰 인덱스
        self.trg_sos_idx = trg_sos_idx  # 타겟 문장의 시작 토큰 인덱스 (SOS token)
        self.device = device
        self.encoder = Encoder(d_model=d_model,  # 모델 사이즈 
                               n_head=n_head,  # Multi-Head Attention의 Head 수
                               max_len=max_len,  # 최대 문장 길이
                               ffn_hidden=ffn_hidden,  # Feed Forward Layer의 Hidden 차원
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,  # Encoder/Decoder 레이어 수
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)  
        trg_mask = self.make_trg_mask(trg) 
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):  # 패딩 토큰이 연산에 영향을 미치지 않도록 마스크 
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):  # 미래 토큰 참조 방지
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask