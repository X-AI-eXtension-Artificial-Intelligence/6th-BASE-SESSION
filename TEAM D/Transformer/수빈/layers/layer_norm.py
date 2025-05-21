"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
'''
LayerNorm은 Transformer 논문에서 사용된 Layer Normalization을 구현한 코드.

레이어 정규화:
- 입력 벡터의 평균과 분산을 이용해 정규화 (스케일을 맞춤)
+ 각 벡터(단어 임베딩)에 대해 평균과 분산 계산
+ (입력 - 평균) / (표준편차) 로 정규화
+ 학습 가능한 scale (gamma), shift (beta) 적용

-> 값이 너무 커지거나 작아지는 걸 방지
-> 학습 안정화와 수렴에 도움
-> Transformer에서는 각 sublayer의 출력마다 적용
'''
'''
입력 x
 └→ mean 계산
 └→ var 계산
 └→ (x - mean) / sqrt(var + eps)
 └→ gamma * (정규화값) + beta
 └→ 출력
'''

import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Transformer에서 사용되는 Layer Normalization 구현
    - 입력 벡터의 마지막 차원을 기준으로 정규화
    - (x - mean) / sqrt(var + eps) 공식
    """
    def __init__(self, d_model, eps=1e-12):
        """
        초기화 함수

        :param d_model: 임베딩 차원 (feature dimension)
        :param eps: 분모가 0이 되는 걸 방지하는 작은 수
        """
        super(LayerNorm, self).__init__()
        # 학습 가능한 scale (gamma)와 shift (beta) 파라미터
        # shape: [d_model]
        self.gamma = nn.Parameter(torch.ones(d_model)) # 초기값 1
        self.beta = nn.Parameter(torch.zeros(d_model)) # 초기값 0
        self.eps = eps # 작은 수 (0으로 나누기 방지용)

    def forward(self, x):
        """
        forward 연산
        입력 벡터 x에 layer normalization 적용

        :param x: 입력 텐서 (예: [batch_size, seq_len, d_model])
        :return: 정규화된 출력 텐서 (동일 shape)
        """

        # 입력 텐서의 마지막 차원 기준 평균 계산
        mean = x.mean(-1, keepdim=True)     # [batch_size, seq_len, 1]

        # 입력 텐서의 마지막 차원 기준 분산 계산
        var = x.var(-1, unbiased=False, keepdim=True)# [batch_size, seq_len, 1]

        # 평균과 분산을 이용해 정규화
        # '-1' means last dimension. 
        out = (x - mean) / torch.sqrt(var + self.eps) # (x - mean) / stddev

        # scale (gamma)와 shift (beta) 적용
        out = self.gamma * out + self.beta # 최종 출력
        return out
