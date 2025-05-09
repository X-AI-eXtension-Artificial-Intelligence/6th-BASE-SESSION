"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

# 위치 인코딩과 토큰 임베딩 모듈 불러오기
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    토큰 임베딩과 포지셔널 인코딩을 결합한 임베딩 클래스
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary (어휘 수)
        :param d_model: dimensions of model (임베딩 차원)
        :param max_len: 시퀀스 최대 길이
        :param drop_prob: 드롭아웃 확률
        :param device: 연산 디바이스 (cuda 또는 cpu)
        """
        super(TransformerEmbedding, self).__init__()

        # 토큰 임베딩: 단어 인덱스를 고정 차원의 벡터로 변환
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 포지션 인코딩: 각 위치에 고유한 정보를 부여 (사인/코사인 기반)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # 드롭아웃: 과적합 방지
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 입력 x: [batch_size, seq_len] (토큰 인덱스)
        tok_emb = self.tok_emb(x)         # shape: [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x)         # shape: [seq_len, d_model]

        # 위치 인코딩을 각 배치에 브로드캐스팅하여 더한 뒤 드롭아웃
        return self.drop_out(tok_emb + pos_emb)
        # 최종 출력 shape: [batch_size, seq_len, d_model]
