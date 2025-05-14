from torch import nn


#nn.Embedding 상속으로 파이토치 내장 임베딩 레이어 사용 
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model): #단어 사전 어휘 크기, 모델 차원
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1) #padding_idx = 1, 패딩 토큰의 값을 1로 인식식하고 학습 안함
        
        