import torch.nn as nn
from .residual import ResidualConnection
from .attention import MultiHeadAttentionBlock
from .feedforward import FeedForwardBlock
from .normalization import LayerNormalization

# 인코더 내부 한 블록 정의하는 클래스
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block # 셀프 어텐션 블록
        self.feed_forward_block = feed_forward_block # 피드포워드 블록
        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout),
            ResidualConnection(features, dropout)
        ])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# 인코더 전체를 정의하는 클래스
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # 여러 개의 인코더 블록으로 구성됨
        self.norm = LayerNormalization(features) # 마지막에 정규화 적용

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
