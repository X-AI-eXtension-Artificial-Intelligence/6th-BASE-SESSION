import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


#만든 인코더, 디코더로
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx #소스 문장 패딩 인덱스
        self.trg_pad_idx = trg_pad_idx #타겟 문장 패딩 인덱스
        self.trg_sos_idx = trg_sos_idx #타겟 문장 sos 인덱스
        self.device = device

        #인코더 디코더 구성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
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
        src_mask = self.make_src_mask(src) #소스 문장 마스크 생성
        trg_mask = self.make_trg_mask(trg) #타겟 문장 마스크 생성
        enc_src = self.encoder(src, src_mask) # 인코더 출력 생성
        output = self.decoder(trg, enc_src, trg_mask, src_mask) #디코더 output 생성 
        return output

    #Attention에서 패딩인 부분은 0, 패딩이 아닌 부분은 1, 차원 맞춰주기
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    #Target 마스크
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device) #하삼각 행렬로 다음 단어 못보게
        trg_mask = trg_pad_mask & trg_sub_mask # 두개 마스크 합치기
        return trg_mask