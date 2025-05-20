import torch
from torch import nn


class LayerNorm(nn.Module): # 레이어 정규화 클래스 정의
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # 정규화된 값에 곱할 학습 가능한 스케일 파라미터 생성
        self.beta = nn.Parameter(torch.zeros(d_model)) # 정규화된 값에 더할 학습 가능한 이동 파라미터 생성
        self.eps = eps # 분모가 0이 되는 것을 방지하기 위한 작은 값

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 마지막 차원을 기준으로 평균 계산
        var = x.var(-1, unbiased=False, keepdim=True) # 마지막 차원을 기준으로 분산 계산
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps) # 입력에서 평균을 빼고 표준편차로 나눠 정규화
        out = self.gamma * out + self.beta # 정규화된 값에 스케일과 이동 파라미터 적용
        return out # 최종 출력 반환
