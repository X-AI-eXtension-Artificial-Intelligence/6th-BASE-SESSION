import torch
import torch.nn as nn

# 레이어 정규화를 수행하는 클래스
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps # 작은 상수로 0으로 나누는 것을 방지
        self.alpha = nn.Parameter(torch.ones(features)) # 정규화된 값에 곱할 학습 가능한 스케일 파라미터
        self.bias = nn.Parameter(torch.zeros(features)) # 정규화된 값에 더할 학습 가능한 바이어스 파라미터

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # 마지막 차원 기준 평균값 계산
        std = x.std(dim=-1, keepdim=True) # 마지막 차원 기준 표준편차 계산
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
