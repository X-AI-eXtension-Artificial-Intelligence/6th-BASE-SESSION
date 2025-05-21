import torch
from torch import nn

# positional encoding
class PositionalEncoding(nn.Module):
    # 사인/코사인 함수를 이용한 포지셔널 인코딩 계산 클래스. 
    # 입력 시퀀스의 각 위치에 대해 고유한 위치 정보를 부여함.

    def __init__(self, d_model, max_len, device):
        # d_model: 임베딩 차원(모델 차원)
        # max_len: 최대 시퀀스 길이
        # device: 연산에 사용할 디바이스(cpu/gpu)

        super(PositionalEncoding, self).__init__()

        # 인코딩 행렬 생성 (모든 위치와 임베딩 차원에 대해)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 포지셔널 인코딩은 학습하지 않음

        # 각 위치(pos)에 대한 인덱스 벡터 생성: (max_len, 1)
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze (단어의 위치를 나타내기 위해)

        # 각 임베딩 차원의 인덱스 벡터 생성(짝수만): (d_model/2,)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i'는 d_model의 인덱스 (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2"는 'i'가 2씩 증가한다는 뜻 (same with 2 * i)

        # 짝수 인덱스(2i)에는 sin, 홀수 인덱스(2i+1)에는 cos 적용
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # (max_len, d_model) 크기의 포지셔널 인코딩 완성
        # 단어의 위치 정보를 고려하기 위해 포지셔널 인코딩을 계산함

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        # x: 입력 텐서 (batch_size, seq_len)
        # return: 포지셔널 인코딩 (seq_len, d_model)

        # [batch_size = 128, seq_len = 30]
        # 입력 시퀀스 길이에 맞는 포지셔널 인코딩 슬라이스 반환
        return self.encoding[:x.size(1), :]
        # [seq_len = 30, d_model = 512]
        # 이것이 tok_emb와 더해짐 -> [128, 30, 512]
        # (seq_len, d_model) 크기 반환 -> 이후 브로드캐스팅으로 배치에 더해짐

# token_embeddings
class TokenEmbedding(nn.Embedding):
    # 토큰 임베딩 클래스. 
    # 단어 인덱스를 임베딩 벡터로 변환 (가중치 행렬을 학습함).

    def __init__(self, vocab_size, d_model):
        # vocab_size: 단어장 크기
        # d_model: 임베딩 차원(모델 차원)

        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        # nn.Embedding 상속. padding_idx=1은 패딩 토큰 인덱스가 1임을 의미

# transformer embedding
class TransformerEmbedding(nn.Module):
    # 토큰 임베딩 + 포지셔널 인코딩 + 드롭아웃을 결합한 임베딩 레이어.
    # 트랜스포머 입력에 위치 정보를 더해줌.

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        # vocab_size: 단어장 크기
        # d_model: 임베딩 차원(모델 차원)
        # max_len: 최대 시퀀스 길이
        # drop_prob: 드롭아웃 확률
        # device: 연산에 사용할 디바이스(cpu/gpu)

        super(TransformerEmbedding, self).__init__()

        # 1. 토큰 임베딩 레이어
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        # 2. 포지셔널 인코딩 레이어
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        # 3. 드롭아웃 레이어
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 입력 시퀀스에 대해 (토큰 임베딩 + 포지셔널 인코딩) 후 드롭아웃 적용
        # x: 입력 텐서 (batch_size, seq_len)
        # return: 임베딩 결과 (batch_size, seq_len, d_model)

        # 1. 토큰 임베딩: (batch_size, seq_len, d_model)
        tok_emb = self.tok_emb(x)
        # 2. 포지셔널 인코딩: (seq_len, d_model)
        pos_emb = self.pos_emb(x)
        # 3. 두 임베딩을 더하고 드롭아웃 적용
        return self.drop_out(tok_emb + pos_emb)
