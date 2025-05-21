"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
'''
PositionalEncoding
: 각 단어에 "너는 문장에서 몇 번째 단어야" 라는 위치 정보를 벡터로 추가해주는 클래스

- 입력이 그냥 단어 index라면 → 위치 정보가 없음
- Transformer는 RNN처럼 순서 기반이 아니라서 → 따로 위치 정보를 인코딩해야 함
    → sine/cosine 공식으로 각 위치마다 unique한 벡터 생성
'''
'''
입력: d_model, max_len
 └→ (pos, dimension index)로 sine/cosine 계산
     └→ [max_len, d_model] 크기의 encoding matrix 생성
         └→ forward(x): 입력 시퀀스 길이만큼 encoding 잘라서 반환
'''
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding 계산 클래스
    - 논문에서 제안한 sine/cosine 기반 위치 인코딩 계산
    """

    def __init__(self, d_model, max_len=512, device='cpu'):
        """
        초기화 함수

        :param d_model: 모델 차원 (예: 512)
        :param max_len: 시퀀스 최대 길이
        :param device: 연산 장치 (cpu or cuda)
        """
        super(PositionalEncoding, self).__init__()

        # 1. 인코딩을 담을 matrix 생성 (max_len x d_model)        
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        # 2. 위치 인덱스 생성 (0 ~ max_len-1)
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        # 3. d_model의 짝수 index [0, 2, 4, ..., d_model-2]
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        # 4. 각 position과 dimension 조합에 대해 sine/cosine 계산
        # sin → even index, cos → odd index
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        """
        forward 함수
        - 입력된 문장의 길이에 맞게 positional encoding 반환

        :param x: 입력 벡터 (batch_size x seq_len)
        :return: positional encoding (seq_len x d_model)
        """
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size() # x는 [batch_size, seq_len] (word index)

        # [batch_size = 128, seq_len = 30]

        # 인코딩 matrix에서 필요한 부분 (seq_len 만큼) 잘라 반환
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
        # 반환 shape: [seq_len, d_model]
