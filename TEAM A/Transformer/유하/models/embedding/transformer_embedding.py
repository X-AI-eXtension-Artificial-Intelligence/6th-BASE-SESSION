from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    -> 해당 클래스가 토큰 임베딩과 위치 인코딩을 결합함을 설명
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model) # 토큰 임베딩 모듈 생성
        self.pos_emb = PositionalEncoding(d_model, max_len, device) # 위치 인코딩 모듈 생성
        self.drop_out = nn.Dropout(p=drop_prob) # 드롭아웃 레이어 생성

    def forward(self, x):
        tok_emb = self.tok_emb(x) # 입력 시퀀스 -> 토큰 임베딩 벡터 변환
        pos_emb = self.pos_emb(x) # 입력 시퀀스 길이에 맞는 위치 인코딩 생성
        return self.drop_out(tok_emb + pos_emb) # 토큰 임베딩 + 위치 인코딩 -> 드롭아웃 적용한 결과 반환
