import torch
from torch import nn


# 사인과 코사인을 활용한 포지셔널 인코딩
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device): #모델 차원, 최대 시퀀스, 장치 파라미터로 받아오기
        
        super(PositionalEncoding, self).__init__()

        # 인풋에 더해주기 위해서 (최대 시퀀스 길이 * 모델 차원) 크기의 2D 텐서 생성(0으로 채워져있는는)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 포지셔널 인코딩은 학습 안함함

        pos = torch.arange(0, max_len, device=device) #0부터 시퀀스 최대길이 만큼의 위치 정보 생성
        pos = pos.float().unsqueeze(dim=1) #unsqueeze로 (시퀀스 최대길이, 1)의 2D 텐서로 위치 정보 생성
        # Sin,Cos 연산에 부동소수점 필요하기 때문에 float로 변환
        
        # 0부터 차례대로 짝수 인덱스만 뽑아와서 float형태로 변수에 저장
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 인덱스에 sin, 홀수 인덱스에 cos 적용용
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    #input x 받았을 때 포지셔널 인코딩 반환하는 함수
    def forward(self, x):

        batch_size, seq_len = x.size() #입력 텐서 (배치크기, 시퀀스 길이)

        return self.encoding[:seq_len, :] #전체 위치 인코딩 반환

