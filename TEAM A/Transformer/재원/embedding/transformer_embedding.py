from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


# 포지셔널 인코딩이랑 기존 임베딩 더해서 최종 트랜스포머 인코더에 들어갈 임베딩 생성성
class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):

        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model) #토큰 임베딩
        self.pos_emb = PositionalEncoding(d_model, max_len, device) #포지셔널 인코딩 정보
        self.drop_out = nn.Dropout(p=drop_prob) #dropout 비율 지정정

    def forward(self, x):
        tok_emb = self.tok_emb(x) 
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb) # 두 개 임베딩 더해주기
