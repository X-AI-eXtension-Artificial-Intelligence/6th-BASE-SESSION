# TokenEmbedding + PositionalEncoding + Dropout을 결합한 Transformer 입력 준비 레이어
from torch import nn  

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
   

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        :param vocab_size: 단어 사전 크기
        :param d_model: 임베딩 차원 (Transformer 모델 차원)
        :param max_len: 위치 인코딩을 위한 최대 시퀀스 길이
        :param drop_prob: 드롭아웃 확률
        :param device: 연산을 수행할 장치 (CPU or GPU)
        """
        super(TransformerEmbedding, self).__init__()

        # 토큰 인덱스를 d_model 차원의 벡터로 변환하는 임베딩 레이어
        self.tok_emb = TokenEmbedding(vocab_size, d_model)

        # 위치 정보를 부여하는 sinusoidal positional encoding
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # 과적합 방지를 위한 드롭아웃 레이어
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # x: [batch_size, seq_len] 형태의 입력 (토큰 인덱스)

        tok_emb = self.tok_emb(x)       # [batch_size, seq_len, d_model] — 단어 임베딩
        pos_emb = self.pos_emb(x)       # [seq_len, d_model] — 위치 인코딩 (broadcast 예정)

        # 위치 인코딩은 [1, seq_len, d_model]로 broadcast되어 더해짐
        return self.drop_out(tok_emb + pos_emb)  # 최종 임베딩에 드롭아웃 적용 후 반환
