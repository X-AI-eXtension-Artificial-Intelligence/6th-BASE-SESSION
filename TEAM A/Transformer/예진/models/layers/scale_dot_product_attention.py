"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention 모듈

    입력: Query, Key, Value
    출력: Attention 결과 벡터 + Attention 가중치(score)

    계산 수식: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

        # softmax: 마지막 차원(-1) 기준으로 수행
        # 각 Query 위치에서 Key 전체에 대해 attention 분포 생성
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        q, k, v: 각각 Query, Key, Value 텐서
        e: 작은 수, 계산 안정성을 위한 epsilon
        """

        # 1. Query와 Key의 내적을 계산하여 유사도 점수 생성
        #    Q @ K^T 를 위해 Key를 전치함
        k_t = k.transpose(2, 3)  
        score = (q @ k_t) / math.sqrt(k.size(-1))  # 내적 후 sqrt(d_k)로 나눔 (스케일 조정)

        # 2. 마스크 적용 (필요한 경우)
        #    마스크가 0인 위치는 -10000으로 만들어 softmax 시 거의 0에 가깝게 만듦
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. softmax로 attention weight 계산
        #    각 Query 위치에서 어떤 Key에 집중할지 확률로 나타냄
        score = self.softmax(score)

        # 4. attention weight를 Value에 곱해 최종 context vector 생성
        #    유사한 단어일수록 더 큰 가중치
        v = score @ v

        return v, score
