import torch.nn as nn

# 전체 Transformer 모델 클래스
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer) -> None:
        super().__init__()
        self.encoder = encoder # 인코더 모듈
        self.decoder = decoder # 디코더 모듈
        self.src_embed = src_embed # 소스 임베딩
        self.tgt_embed = tgt_embed # 타겟 임베딩
        self.src_pos = src_pos # 소스 위치 인코딩
        self.tgt_pos = tgt_pos # 타겟 위치 인코딩
        self.projection_layer = projection_layer # 출력 레이어

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
