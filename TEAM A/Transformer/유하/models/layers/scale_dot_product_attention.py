import math

from torch import nn


class ScaleDotProductAttention(nn.Module): 
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # 마지막 차원에 대해 소프트맥스 함수 적용하는 객체 생성

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size() # key 텐서의 배치 크기, 헤드 수, 시퀀스 길이, 임베딩 차원 가져옴

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # key 텐서의 마지막 두 차원을 전치
        score = (q @ k_t) / math.sqrt(d_tensor)  # query와 전치된 key의 행렬곱 계산 후, 임베딩 차원의 제곱근으로 나눠 스케일링

        # 2. apply masking (opt)
        if mask is not None: # 마스크가 있으면, 마스크가 0인 위치에 매우 작은 값을 넣어 소프트맥스에서 무시되도록 함 
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score) # 소프트맥스 적용 -> 어텐션 확률로 변환

        # 4. multiply with Value
        v = score @ v # 어텐션 점수와 value 벡터를 곱해 최종 출력

        return v, score # 최종 출력과 어텐션 확률 반환
