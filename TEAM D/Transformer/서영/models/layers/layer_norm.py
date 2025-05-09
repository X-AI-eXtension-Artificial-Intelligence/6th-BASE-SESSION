"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    # Transformer에 사용되는 Layer Normalization 구현
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # 학습 가능한 스케일 파라미터 γ (초기값: 1)
        self.gamma = nn.Parameter(torch.ones(d_model))

        # 학습 가능한 이동 파라미터 β (초기값: 0)
        self.beta = nn.Parameter(torch.zeros(d_model))

        # 분모 안정화를 위한 작은 상수 ε
        self.eps = eps

    def forward(self, x):
        # 입력 텐서 x의 마지막 차원 기준으로 평균과 분산 계산
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        # keepdim=True: 차원을 유지해 브로드캐스팅이 가능하도록 함

        # 정규화: 평균을 빼고 분산으로 나눈 뒤, 스케일 및 이동
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta  # 학습 가능한 스케일과 이동 적용

        return out  # shape 동일: [batch_size, seq_len, d_model]
