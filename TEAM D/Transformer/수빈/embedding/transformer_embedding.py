"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
'''
TransformerEmbedding
: 단어 임베딩 + 위치 인코딩 → dropout → 출력” 역할.

-> 단어 임베딩 (TokenEmbedding)
-> 위치 인코딩 (PositionalEncoding)
두 가지를 더해서 출력하는 역할.

Transformer는 입력으로 단어의 순서를 모르기 때문에
→ 단어 벡터 + 위치 정보 벡터를 더해줘야 함
Dropout으로 과적합 방지
'''
'''
입력 (단어 인덱스)
 └→ TokenEmbedding
 └→ PositionalEncoding
     └→ TokenEmbedding + PositionalEncoding (벡터 덧셈)
         └→ Dropout
             └→ 출력 (임베딩 벡터)
'''
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network

    TransformerEmbedding 클래스
    - TokenEmbedding + PositionalEncoding 합친 embedding layer
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information
        초기화 함수
        :param vocab_size: 단어 사전 크기
        :param d_model: 임베딩 벡터 차원
        :param max_len: 문장의 최대 길이
        :param drop_prob: dropout 비율
        :param device: 연산 장치 (cpu or cuda)
        """

        super(TransformerEmbedding, self).__init__()

        # 1. 단어 임베딩 layer (단어 index → 벡터)
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 2. 위치 인코딩 layer (포지셔널 인코딩 계산)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # 3. dropout layer
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        forward 함수
        - 단어 임베딩 + 위치 인코딩 더해서 dropout 적용

        :param x: 입력 (batch_size x seq_len) → 단어 인덱스
        :return: embedding 출력 (batch_size x seq_len x d_model)
        """
        x = x.to(self.tok_emb.weight.device)
        tok_emb = self.tok_emb(x)               # 단어 인덱스를 벡터로 변환 (token embedding)
        pos_emb = self.pos_emb(x)               # 위치 인코딩 벡터 계산
        return self.drop_out(tok_emb + pos_emb) # 두 벡터 더하고 dropout