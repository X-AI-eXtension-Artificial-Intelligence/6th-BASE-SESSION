"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

from torch import nn

# 단어 임베딩 모듈 (nn.Embedding 기반)
from models.embedding.token_embeddings import TokenEmbedding

# 위치 정보를 담는 사인/코사인 기반 위치 인코딩
from models.embedding.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    TransformerEmbedding 클래스:
    - TokenEmbedding(단어 임베딩) + PositionalEncoding(위치 인코딩) 합침
    - 단어의 의미 + 순서 정보를 함께 담은 벡터를 생성
    - Transformer 입력 전처리 단계
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        vocab_size: 전체 단어 크기
        d_model: 임베딩 벡터 차원 수
        max_len: 문장의 최대 길이 (위치 인코딩 범위)
        drop_prob: 드롭아웃 확률
        device: 연산 장치
        """
        super(TransformerEmbedding, self).__init__()

        # 1. TokenEmbedding 모듈 초기화: 단어 ID를 d_model 차원의 연속 벡터로 변환
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 2. PositionalEncoding 모듈 초기화: 각 위치(0~max_len)에 대해 사인/코사인 위치 벡터 미리 계산
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # 3. 드롭아웃: embedding + position encoding 후에 적용
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        x: 단어 ID 시퀀스
        출력: 각 단어에 대해 의미 + 위치 정보가 더해진 벡터
        """

        # 1. Token Embedding (단어 -> 벡터)
        tok_emb = self.tok_emb(x)  

        # 2. Positional Encoding (위치 -> 벡터)
        pos_emb = self.pos_emb(x)  

        # 3. 두 벡터를 더해 위치 정보를 포함한 임베딩 생성
        out = tok_emb + pos_emb

        # 4. dropout 적용
        return self.drop_out(out)
