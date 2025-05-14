from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    -> 해당 클래스가 nn.Embedding을 활용해 단어를 밀집 벡터로 변환함을 설명
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1) # 부모 클래스 생성자 호출 + 패딩 토큰 인덱스 1로 지정
