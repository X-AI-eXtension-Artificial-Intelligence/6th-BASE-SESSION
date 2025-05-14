import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding. -> 사인/코사인 기반 위치 인코딩 계산
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model 
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device) # (max_len, d_model) 크기의 0으로 채워진 위치 인코딩 행렬 생성
        self.encoding.requires_grad = False  # 위치 인코딩은 학습하지 않으므로, 그래디언트 계산을 비활성화

        pos = torch.arange(0, max_len, device=device) # 0부터 max_len-1까지 위치 인덱스 벡터를 생성
        pos = pos.float().unsqueeze(dim=1) # 위치 벡터를 float 타입으로 변환하고, 2차원(열 벡터)으로 만들어줌
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float() # 임베딩 차원 중 짝수 인덱스를 float 타입으로 생성
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))) # 짝수 인덱스에 대해 사인 함수를 이용해 위치 인코딩 값을 계산
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model))) # 홀수 인덱스에 대해 코사인 함수를 이용해 위치 인코딩 값을 계산
        # compute positional encoding to consider positional information of words

    def forward(self, x): # 순전파 연산
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size() # 입력 x의 배치 크기와 시퀀스 길이입력 x의 배치 크기와 시퀀스 길이
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :] # 입력 시퀀스 길이에 맞는 위치 인코딩 부분만 반환
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512] 
        # -> 최종 임베딩: [배치, 시퀀스길이, 임베딩차원]
