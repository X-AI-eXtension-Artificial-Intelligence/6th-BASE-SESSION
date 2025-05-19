import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        
        ## 소스 문장과 타겟 문장으로 나누어 임베딩
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

        ## 인코더, 디코더 생성
        self.encoder = encoder
        self.decoder = decoder

        ## 출력값을 단어 확률 분포로 변환하는 모듈
        self.generator = generator

    ## 입력 문장을 마스크 씌워서(선택) 디코더로 전달
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    ## 인코더의 출력과 타겟 문장과 어텐션하는 등 계산 후 출력
    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    def forward(self, src, tgt):
        ## 입력문장, 타겟문장 각각 마스킹 생성
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)

        ## 인코더 출력, 디코더 출력
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        
        ## 디코더의 출력을 받아서 실제 단어 확률분포로 변환
        out = self.generator(decoder_out)

        ## 하고 LOG SOFTMAX 까지
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

    ## 입력 문장에 패딩 마스크 생성
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask

    ## 출력 문장 패딩 마스킹 및 뒤 단어 마스킹
    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        return pad_mask & seq_mask

    ## 소스, 타겟 문장 간 어텐션할 때 패딩 토큰이 영향없도록 마스킹
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask

    ## <pad> 토큰을 마스킹
    def make_pad_mask(self, query, key, pad_idx=1):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        ## Key(입력 시퀀스)의 패딩 여부 확인
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (batch_size, 1, query_seq_len, key_seq_len)

        ## Query(출력 시퀀스)의 패딩 여부 확인
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (batch_size, 1, query_seq_len, key_seq_len)

        ## 두 마스크를 결합하여 최종 마스크 생성
        mask = key_mask & query_mask
        mask.requires_grad = False  ## 마스크는 학습하지 않음
        return mask

    ## 뒤 단어는 보지 못하도록 마스킹
    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask