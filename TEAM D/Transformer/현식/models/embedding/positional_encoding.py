import torch
from torch import nn


class PositionalEncoding(nn.Module):

    # d_model : 모델 차원 (ex : 512)
    # max_len : 최대 시퀀스 길이 (ex : 512)
    def __init__(self, d_model, max_len, device):

        # nn.Module의 __init__()을 호출하여 부모 클래스 초기화
        super(PositionalEncoding, self).__init__()

        # 크기가 (max_len x d_model)인 0으로 초기화된 텐서를 생성
        # 이후 sin과 cos 값을 채울 예정
        self.encoding = torch.zeros(max_len, d_model, device=device)

        # Positional Enocding은 학습되지 않고 고정된 값으로 사용
        self.encoding.requires_grad = False  

        # pos (위치값) 생성
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # 차원 인덱스 생성 (짝수 인덱스만 저장)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # Sin, Cos 값 계산
        # 짝수 인덱스에 sin 값 계산
        # 홀수 인덱스에 cos 값 계산산
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]
