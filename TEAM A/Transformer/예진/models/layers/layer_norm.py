"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer Normalization 클래스
    - 입력 벡터의 마지막 차원에 대해 정규화 수행
    - Transformer의 각 블록마다 사용되어 학습 안정성과 수렴 속도 향상에 기여
    """

    def __init__(self, d_model, eps=1e-12):
        """
        d_model: 입력 벡터의 차원 수
        eps: 분모가 0이 되는 걸 방지하기 위한 아주 작은 값 (안정성 목적)
        """
        super(LayerNorm, self).__init__()

        # 가중치 파라미터
        self.gamma = nn.Parameter(torch.ones(d_model))

        # 편향 파라미터
        self.beta = nn.Parameter(torch.zeros(d_model))

        # 분산 계산 시 0으로 나누는 것 방지용
        self.eps = eps

    def forward(self, x):
        """
        x: 입력 텐서 (shape: [batch_size, seq_len, d_model])
        마지막 차원(d_model)에 대해 평균과 분산을 구하고 정규화함
        """

        # 마지막 차원(-1)에 대해 평균 계산 -> 각 토큰 벡터마다 평균 하나씩
        mean = x.mean(-1, keepdim=True)

        # 마지막 차원에 대해 분산 계산
        var = x.var(-1, unbiased=False, keepdim=True)

        # 정규화 수행: (x - 평균) / (표준편차 + epsilon)
        # keepdim=True라 shape 그대로 유지됨
        out = (x - mean) / torch.sqrt(var + self.eps)

        # 정규화된 값에 가중치 + 편향 적용해 다시 스케일 조정
        out = self.gamma * out + self.beta

        return out
