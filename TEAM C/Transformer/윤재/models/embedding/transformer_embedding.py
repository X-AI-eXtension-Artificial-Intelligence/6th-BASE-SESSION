"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)  # [batch, seq_len, d_model]
        pos_emb = self.pos_emb(x)  # [seq_len, d_model]
        emb = tok_emb + pos_emb.unsqueeze(0)  # [batch, seq_len, d_model]
        emb = nn.functional.normalize(emb, p=2, dim=-1)  # L2 정규화
        return self.drop_out(emb)

#### Cosine Similariry Attention을 진행했기 때문에, 벡터 값들이 전부 0~1 사이임
#### -> 벡터 span에서 거리가 아닌 "방향"에 집중 (즉, 모든 vector span을 길이가 1인 vector 구 span으로 응집)
#### -> 이는 의미, 맥락적으로 유사한 토큰들에 더 집중하고 싶어서 그럼 (유사한 토큰을 더 잘 분류)
####    => 동의어/유의어 분류, 동일 단어 의미 맞추기, 미세 감정 분류, 개체명 세분화 등등 task에 효과적
#### 그러나 Encoding 결과가 1보다 크면 모델은 임베딩에 더 잘 집중하기 때문에
#### 맥락을 더 잘 파악하려던 기존의 의도가 빗나갈 수 있기에, 똑같이 정규화해주는 L2 정규화 진행

