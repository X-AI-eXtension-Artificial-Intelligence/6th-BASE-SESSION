"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
'''
TokenEmbedding
: 단어 인덱스를 벡터로 바꿔주는 클래스

-> Embedding 레이어가 학습을 통해 임베딩 행렬(W)을 만들어
-> 입력 인덱스 → 해당하는 벡터를 lookup해서 반환
'''
'''
입력: 단어 인덱스
 └→ Embedding lookup
     └→ 임베딩 벡터 반환
'''
from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """
    """
    Token Embedding 클래스
    - torch.nn.Embedding을 상속받아 작성
    - 단어 인덱스를 dense vector로 변환
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        초기화 함수

        :param vocab_size: 단어 사전 크기 (vocabulary size)
        :param d_model: 임베딩 벡터 차원
        """
        # super()로 torch.nn.Embedding 초기화
        # padding_idx=1: 인덱스 1은 padding으로 간주 (해당 벡터는 gradient 업데이트 X)
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
