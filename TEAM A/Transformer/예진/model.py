
# model.py
# Transformer, Encoder, Decoder 및 모든 하위 블록 통합

import torch
import torch.nn as nn
import math

# -----------------------------
# LayerNorm
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta

# ... (중략) 전체 코드는 canvas에서 사용 가능
