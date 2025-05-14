"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model, #모델 차원 
                                        max_len=max_len, #최대 시퀀스 길이 
                                        vocab_size=enc_voc_size, #입력 사전 크기
                                        drop_prob=drop_prob, #드롭아웃 확률
                                        device=device) 

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)]) #지정한 개수만큼 레이어 반복 생성 

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers: #n_layers 만큼 인코더 레이어 반복 적용
            x = layer(x, src_mask) #각레이어에서 selfattention과 FFN 통과

        return x