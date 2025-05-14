# 단어(토큰)를 임베딩 벡터로 바꾸는 역할을 합니다.


from torch import nn  

class TokenEmbedding(nn.Embedding):
    """
    nn.Embedding을 상속한 TokenEmbedding 클래스
    각 단어(토큰)를 고정된 차원의 밀집 벡터(dense vector)로 변환
    """

    def __init__(self, vocab_size, d_model):
        """
        토큰 임베딩 레이어 생성자

         vocab_size: 전체 단어 사전 크기 (예: 10,000개 단어)
         d_model: 각 단어 임베딩의 차원 (= 모델 차원)
        """
        # nn.Embedding(vocab_size, d_model)을 초기화
        # 인덱스 1번(PAD 토큰)은 학습되지 않도록 설정 (출력은 항상 0)
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

        # 패딩 토큰 자체가 1번이라는 뜻    관례적으로 1번에 둔다.


