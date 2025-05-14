from torch import nn

# 인코더 전체 구성
from models.blocks.encoder_layer import EncoderLayer 
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        #인풋 임베딩 구성
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size, #어휘 사전 구현
                                        drop_prob=drop_prob,
                                        device=device)

        #레이어 구성
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, #hidden state 차원
                                                  ffn_hidden=ffn_hidden, # 설계한 ffn
                                                  n_head=n_head, #헤드수
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)]) # 반복을 통한 순차적 레이어


    def forward(self, x, src_mask):
        x = self.emb(x) #임베딩 얻기

        for layer in self.layers: #모듈리스트에서 반복적인 인코더 실행
            x = layer(x, src_mask) #x와 마스크 함께 전달

        return x