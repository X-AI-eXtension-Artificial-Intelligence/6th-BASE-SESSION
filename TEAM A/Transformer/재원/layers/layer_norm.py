import torch
from torch import nn


# 레이어 정규화 클래스 구현
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        # eps 분산이 0이 되는 것을 방지하기 위한 아주 작은 값
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) #스케일(곱)
        self.beta = nn.Parameter(torch.zeros(d_model)) #이동(더하기)
        self.eps = eps #분산 0이 되는 것을 방지

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) #평균계산
        var = x.var(-1, unbiased=False, keepdim=True) #분산 계산

        out = (x - mean) / torch.sqrt(var + self.eps) #일단 평균 분산 통해서 정규화 하는데, 표현력이 살짝 부족할 수 있어서
        out = self.gamma * out + self.beta # 정규화하고 감마,베타로 스케일 조정, shift 해서 최종 output 생성
        return out
