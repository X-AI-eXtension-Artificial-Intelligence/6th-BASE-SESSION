"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # 포지션 인코딩을 저장할 행렬 생성 (shape: [max_len, d_model])
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 학습되지 않는 고정된 값으로 사용

        # 각 위치 인덱스 생성: (0 ~ max_len-1) → [max_len, 1] 형태로 변형
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # shape: [max_len, 1]

        # 각 임베딩 차원 인덱스 (0, 2, 4, ..., d_model-2)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 인덱스 위치에 대해 사인 함수 적용
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))

        # 홀수 인덱스 위치에 대해 코사인 함수 적용
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 결과적으로 위치에 따라 고유한 주기를 갖는 사인/코사인 값이 생성되어 위치 정보를 부여함

    def forward(self, x):
        # x는 보통 [batch_size, seq_len] 혹은 [batch_size, seq_len, d_model] 형태

        batch_size, seq_len = x.size()
        # 입력 시퀀스 길이만큼 위치 인코딩 반환

        return self.encoding[:seq_len, :]
        # 출력 shape: [seq_len, d_model]
        # 나중에 토큰 임베딩에 broadcasting을 통해 더해져 최종 임베딩이 됨
