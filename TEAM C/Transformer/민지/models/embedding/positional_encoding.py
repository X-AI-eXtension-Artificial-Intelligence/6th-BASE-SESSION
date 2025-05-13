import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # 인풋이랑 똑같은 크기로 pos encoding 매트릭스 생성
        self.encoding = torch.zeros(max_len, d_model, device=device) # Tx512
        # 학습시 반영 안되도록
        self.encoding.requires_grad = False
        # 0~max_len-1까지의 정수 생성
        pos = torch.arange(0, max_len, device=device)
        # 위의 정수를 실수형 텐서로 변환하고, 차원 하나 더 추가 => 2차원 텐서로 변경 (T,) => (T,1)
        pos = pos.float().unsqueeze(dim=1)

        # 짝수 인덱스만 가져옴 - 주기 생성을 위함
        # 각각 torch.sin(x), torch.cos(x)에서 x는 주기를 의미
        # x값이 커질수록 주기가 짧아져 고주파가 됨 => 인덱스가 커질수록 고주파가 되는 것
        # 주기 변화를 점진적으로 만들어 다양한 주파수를 생성하기 위해 이와 같이 주기가 다른 함수를 사용하여 나타내는 것인데,
        # _2i가 홀수까지 포함되면 주파수 간 차이가 너무 미세해져 중복되거나 구분이 어려운 주기가 생성될 수 있음
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, length = x.size()

        return self.encoding[:length, :]


# if __name__ == '__main__':
    # _2i = torch.arange(0, 512, step=2, device='cpu').float()
    # pos = torch.arange(0, 14, device='cpu')
    # pos = pos.float().unsqueeze(dim=1)
    # print((torch.sin(pos / (10000 ** (_2i / 512)))).size())