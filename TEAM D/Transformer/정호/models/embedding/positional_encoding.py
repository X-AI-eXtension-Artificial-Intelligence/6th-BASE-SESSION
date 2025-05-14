"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    
    # 사인/코사인 기반 포지셔널 인코딩을 계산하는 클래스
    # 트랜스포머가 순서를 모르는 문제를 해결하기 위해 위치 정보를 임베딩에 추가가
    

    def __init__(self, d_model, max_len, device):
        
        # 포지셔널 인코딩 클래스 생성자

        # param d_model: 모델의 임베딩 차원
        # param max_len: 인코딩할 최대 시퀀스 길이
        # param device: 연산을 수행할 디바이스 (cpu or cuda)
        
        super(PositionalEncoding, self).__init__()

        # 입력 임베딩과 더해줄 동일 크기의 인코딩 행렬 초기화
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 학습 대상 아님 (고정값)

        # 위치 인덱스 텐서 생성: 0 ~ max_len-1 (열 형태)
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # (max_len, 1) 형태로 변형

        # 짝수 인덱스들: 0, 2, 4, ..., d_model-2
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 적용
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # 짝수
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))  # 홀수

    def forward(self, x):
        
        # 입력 x에 대해 해당하는 길이만큼의 포지셔널 인코딩을 반환

        # param x: 입력 시퀀스 (예: [batch_size, seq_len])
        # return: 위치 인코딩 (shape: [seq_len, d_model])
        
        batch_size, seq_len = x.size()

        # 입력 시퀀스 길이에 맞게 앞부분 슬라이싱하여 반환
        return self.encoding[:seq_len, :]
