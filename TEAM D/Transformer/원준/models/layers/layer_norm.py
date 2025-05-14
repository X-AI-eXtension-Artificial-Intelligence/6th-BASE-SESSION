
import torch
from torch import nn 

class LayerNorm(nn.Module):
    """
    Transformer에서 사용하는 Layer Normalization 클래스
    입력 텐서의 마지막 차원에 대해 정규화하여 학습 안정화
    """

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # 스케일 파라미터 (학습 가능), 초기값은 1
        self.gamma = nn.Parameter(torch.ones(d_model))  
        # d_model: 입력 벡터의 차원 수 (예: 512). 이 차원을 기준으로 정규화를 수행
        # eps: 분산이 0이 될 경우를 막기 위한 작은 수

        # 이동 파라미터 (학습 가능), 초기값은 0
        self.beta = nn.Parameter(torch.zeros(d_model))

        # 감마, 베타 정규화된 값을 조정하기 위한 것 


        self.eps = eps

    def forward(self, x):
        # 입력 텐서의 마지막 차원을 기준으로 평균을 계산 (keepdim=True로 차원 유지)
        mean = x.mean(-1, keepdim=True)

        # 마지막 차원이란?
        # x.shape = [batch_size, seq_len, d_model]   
        # d_model  각 단어   -   512 차원 벡터 정규화 


        # 입력 텐서의 마지막 차원을 기준으로 분산을 계산 (N으로 나누므로 unbiased=False)
        var = x.var(-1, unbiased=False, keepdim=True)

        # 왜 편향된 걸 하느냐? 계산이 수월하고, 정확성보다 안정성을 원하기에 굳이 복잡하게 n-1할 필요가 없다. 

        # 정규화: 평균을 빼고, 분산의 제곱근(표준편차)로 나누기
        out = (x - mean) / torch.sqrt(var + self.eps)

        # 정규화된 출력에 학습 가능한 스케일(gamma)과 이동(beta) 적용
        out = self.gamma * out + self.beta

        return out  # 정규화된 결과 반환
