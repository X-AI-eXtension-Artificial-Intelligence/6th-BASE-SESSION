import torch
from torch import nn 

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding을 계산하는 클래스스
    Transformer는 순서를 모르기 때문에 위치 정보를 임베딩 벡터에 추가하는 것 
    """

    def __init__(self, d_model, max_len, device):
        """
         d_model: 모델 임베딩 차원 크기   # 하나의 단어를 몇 차원짜리 벡터로 표현할까?
         max_len: 처리 가능한 최대 시퀀스 길이  # 모델이 처리할 수 있는 문장의 최대 길이를 미리 정해두는 값
         device: 연산을 수행할 장치 (CPU or GPU)
        """
        super(PositionalEncoding, self).__init__()

        # 위치 임베딩을 저장할 텐서를 0으로 초기화 (shape: [max_len, d_model])
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 학습 대상 아님 (고정된 수식이므로)

        # 각 위치를 나타내는 인덱스: [0, 1, 2, ..., max_len - 1]
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)  # shape: [max_len, 1]로 변환 (브로드캐스팅용)
        # 하나의 벡터로 바꾸기 

        # 짝수 인덱스를 위한 스케일 인덱스: [0, 2, 4, ..., d_model-2]
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        # 짝수 인덱스에 대해 sin 파형 적용
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))

        # 홀수 인덱스에 대해 cos 파형 적용
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 결과적으로 각 위치마다 고유한 sin/cos 기반 패턴을 가짐

    def forward(self, x):
        # 입력 x: 보통 shape = [batch_size, seq_len]

        batch_size, seq_len = x.size()  # 배치 크기와 시퀀스 길이 추출

        return self.encoding[:seq_len, :]  # 해당 시퀀스 길이만큼 잘라서 반환,  뒤는 모델 개수만큼 
