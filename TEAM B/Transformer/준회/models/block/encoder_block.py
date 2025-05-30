import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer

## Residual Connection(attention + FFN) 으로 구성된 인코더 블록
class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=1, gated=True):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate, gated=gated)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate, gated=gated)

    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        return out