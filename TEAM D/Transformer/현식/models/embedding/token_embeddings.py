from torch import nn

# 단어을 벡터(d_model 크기)로 변환
# padding_idx=1이면 토큰 ID가 1인 패딩 토큰은 자동으로 0으로 임베딩됨
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
