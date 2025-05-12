"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding 클래스 (sin/cos 기반 위치 인코딩)

    Transformer는 RNN처럼 순차적 구조가 없기 때문에
    입력 시퀀스에 '단어의 순서' 정보를 추가함
    => 각 위치(position)에 고유한 벡터를 더해주는 방식 사용
    """

    def __init__(self, d_model, max_len, device):
        """
        d_model: 임베딩 차원 수
        max_len: 처리할 최대 문장 길이
        """
        super(PositionalEncoding, self).__init__()

        # 1. 위치 인코딩 저장할 텐서
        self.encoding = torch.zeros(max_len, d_model, device=device)

        # 2. 학습 대상이 아니므로 gradient 계산 안 함
        self.encoding.requires_grad = False

        # 3. 위치 정보 텐서 생성: 0 ~ max_len-1까지 정수값
        # shape: [max_len, 1]
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)

        # 4. 주파수를 조절할 인덱스 텐서 생성
        # 짝수 인덱스 (0, 2, 4, ...)에만 사용됨
        """
        sin 쓸 위치(짝수 차원)만 골라서 계산해두면
        cos는 그 옆자리(홀수 차원)에 자동으로 들어가니까
        따로 cos용 인덱스를 만들 필요 X
        """
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 5. 실제 인코딩 계산
        # 각 위치 pos에 대해 짝수 index: sin, 홀수 index: cos 함수 적용
        # 분모: 주파수를 점점 줄이기 위한 스케일링 -> 10000^(2i / d_model)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 짝수 위치
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 홀수 위치

        # 결과적으로 각 위치마다 고유한 d_model 차원의 벡터가 생성됨

    def forward(self, x):
        """
        x: 입력 텐서 (예: 토큰 임베딩) 
        출력: 입력 길이에 맞는 위치 인코딩 
        """

        # 입력 텐서의 시퀀스 길이 추출
        batch_size, seq_len = x.size()

        # 저장된 인코딩 중 앞쪽 seq_len만 잘라서 사용
        return self.encoding[:seq_len, :]
