import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


# 디코더 구성
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        #트랜스포머 임베딩
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        #레이어개수만큼 디코더 레이어 반복
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        #모델 차원 -> 단어 사전 크기로 변경
        self.linear = nn.Linear(d_model, dec_voc_size)


    def forward(self, trg, enc_src, trg_mask, src_mask):

        #타겟 문장 임베딩으로 변환
        trg = self.emb(trg)

        #시점별로 타겟 마스크, 패딩 마스크 적용
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        #단어 사전 크기의 벡터로
        output = self.linear(trg)
        return output
    
    # 아직 softmax는 미적용