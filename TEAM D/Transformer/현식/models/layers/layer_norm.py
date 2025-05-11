import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # 학습 가능한 γ (기본값 1) 
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 학습 가능한 β (기본값 0)
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 마지막 차원의 평균 계산 (d_model)
        var = x.var(-1, unbiased=False, keepdim=True)  # 마지막 차원의 분산 계산 (d_model)

        out = (x - mean) / torch.sqrt(var + self.eps)  # 정규화 수행
        out = self.gamma * out + self.beta  # γ와 β적용
        return out