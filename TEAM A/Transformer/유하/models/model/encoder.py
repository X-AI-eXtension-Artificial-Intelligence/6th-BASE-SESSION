from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 입력 시퀀스에 대한 임베딩(토큰+위치) 생성
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 지정한 수만큼 인코더 레이어를 쌓아 모듈 리스트로 만들어줌
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x) # 입력 시퀀스에 임베딩 적용

        for layer in self.layers: # 각 인코더 레이어 순차적으로 적용 
            x = layer(x, src_mask)

        return x # 인코더의 최종 출력 반환