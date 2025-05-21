"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
'''
ScaleDotProductAttention
: attention 점수(유사도)를 계산하고 결과를 만들어주는 함수

흐름 순서:
1. Query와 Key 내적 → 어떤 단어끼리 관련 있는지 계산
2. √(key 차원)으로 나누어 → 값 스케일 조정
3. softmax → 유사도를 0~1 확률값으로 변환
4. Value와 곱해서 → 중요한 정보만 모아줌
'''
'''
입력: Query (Q), Key (K), Value (V)
 └→ 내적 (Q × K^T)
     └→ 스케일 (나누기 sqrt(d_k))
         └→ 마스크 적용 (opt)
             └→ softmax
                 └→ 가중합 (softmax score × V)
                     └→ 출력
'''
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention 계산
    - Query와 Key의 내적을 통해 유사도 계산
    - softmax로 가중치화
    - Value에 가중치를 곱해 최종 attention 출력

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        """
        초기화 함수
        - Softmax layer 생성
        """
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)               # 마지막 차원 기준 softmax

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        forward 함수
        - Query, Key, Value로 attention 계산

        :param q: Query 벡터 (batch_size x head x seq_len_q x d_tensor)
        :param k: Key 벡터 (batch_size x head x seq_len_k x d_tensor)
        :param v: Value 벡터 (batch_size x head x seq_len_v x d_tensor)
        :param mask: attention mask (선택적)
        :param e: epsilon (아주 작은 값, 나눗셈 방지)
        :return: attention 결과 v, attention score
        """
        # input is 4 dimension tensor
        # 입력 shape: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. Key를 transpose (마지막 두 차원 swap: seq_len x d_tensor → d_tensor x seq_len)
        k_t = k.transpose(2, 3)  # transpose  # shape: (batch_size, head, d_tensor, seq_len)

        # 2. Query와 Key^T 내적
        # → 유사도(score) 계산: (batch_size, head, seq_len, seq_len)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
                                                 # sqrt(d_k)로 스케일링

        # 3️. (선택적) mask 적용: 마스킹된 부분 -10000으로 만들어 softmax 이후 0되도록
        if mask is not None:
            mask = mask.to(score.device)
            score = score.masked_fill(mask == 0, -10000)

        # 4. softmax 적용 → attention weight (0~1)
        score = self.softmax(score)

        # 5. attention weight와 Value 곱함 → 최종 attention 출력
        v = score @ v  # shape: (batch_size, head, seq_len, d_tensor)

        return v, score
