import torch
from torch import nn

from models.blocks import DecoderLayer, EncoderLayer
from models.embedding import TransformerEmbedding

# decoder
# 입력 시퀀스를 받아 여러 디코더 레이어를 통과시켜 최종적으로 단어 분포를 출력
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        
        # 1. 임베딩 레이어 (토큰 임베딩 + 포지셔널 인코딩)
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        # 2. 디코더 레이어 여러 개 쌓기
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        # 3. 마지막 선형 변환 (출력: 단어장 크기)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: 디코더 입력 (batch, trg_seq_len)
        # enc_src: 인코더 출력 (batch, src_seq_len, d_model)
        # trg_mask: 디코더 마스크 (마스킹된 셀프 어텐션)
        # src_mask: 인코더-디코더 어텐션 마스크
        
        # 1. 임베딩(토큰 + 위치)
        trg = self.emb(trg)

        # 2. 여러 디코더 레이어 통과
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 3. 마지막 선형 변환 (단어장 크기로)
        output = self.linear(trg)
        return output
    
# encoder
# 입력 시퀀스를 받아 여러 인코더 레이어를 통과시켜 인코딩된 시퀀스를 출력
class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        # 1. 임베딩 레이어 (토큰 임베딩 + 포지셔널 인코딩)
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)


        # 2. 인코더 레이어 여러 개 쌓기
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        # x: 인코더 입력 (batch, src_seq_len)
        # src_mask: 소스 마스크 (패딩 마스킹 등)
        
        # 1. 임베딩(토큰 + 위치)
        x = self.emb(x)

        # 2. 여러 인코더 레이어 통과
        for layer in self.layers:
            x = layer(x, src_mask)

        return x

# transformer
# 트랜스포머 전체 모델 (인코더 + 디코더 + 마스킹)
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        # 특수 토큰 인덱스 및 디바이스 저장
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        # 인코더 생성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        # 디코더 생성
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        # src: 인코더 입력 (batch, src_seq_len)
        # trg: 디코더 입력 (batch, trg_seq_len)

        # 1. 소스 마스크 생성 (패딩 위치 마스킹)
        src_mask = self.make_src_mask(src)
        # 2. 타겟 마스크 생성 (패딩+미래 마스킹)
        trg_mask = self.make_trg_mask(trg)
        # 3. 인코더 통과
        enc_src = self.encoder(src, src_mask)
        # 4. 디코더 통과
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        # 소스(입력) 마스크 생성: 패딩 위치를 0, 나머지는 1로 마스킹
        # shape: (batch, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # 타겟(출력) 마스크 생성: 
        # - 패딩 위치 마스킹 + 미래 정보 마스킹(디코더 셀프 어텐션용)
        # shape: (batch, 1, trg_seq_len, trg_seq_len)

        # 1. 패딩 마스크: (batch, 1, 1, trg_seq_len)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # 2. 미래 정보 마스킹: (trg_seq_len, trg_seq_len) 하삼각 행렬
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=self.device)).bool()
        # 3. 두 마스크를 AND 연산으로 결합
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask