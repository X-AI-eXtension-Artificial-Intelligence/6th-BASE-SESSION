"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""

from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    TokenEmbedding 클래스:
    - 단어(토큰)를 고정된 크기의 벡터로 변환하는 nn.Embedding 상속
    - 입력 시퀀스의 각 단어 ID를 d_model 차원의 연속 벡터로 변환
    - Transformer에서는 이 벡터에 positional encoding을 더해 사용

    ex. "나는 밥을" -> [102, 305, 12] -> [3, d_model]짜리 임베딩 벡터로 변환
    """

    def __init__(self, vocab_size, d_model):
        """
        vocab_size: 전체 단어 집합의 크기
        d_model: 임베딩 벡터의 차원
        """

        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)  # nn.Embedding(vocab_size, d_model) 초기화
                                                                                  # padding_idx=1: 단어 ID 1은 padding 용도
                                                                                  #                이 위치는 학습 X
