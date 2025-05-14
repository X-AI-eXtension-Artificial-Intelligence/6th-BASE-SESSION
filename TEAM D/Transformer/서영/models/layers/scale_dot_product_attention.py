"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query (encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # 마지막 차원 기준 softmax

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input은 모두 4차원 텐서: [batch_size, head, length, d_tensor]

        batch_size, head, length, d_tensor = k.size()

        # 1. Key를 transpose하여 Query와 내적 → 유사도 스코어 행렬 계산
        k_t = k.transpose(2, 3)  # shape: [batch_size, head, d_tensor, length]
        score = (q @ k_t) / math.sqrt(d_tensor)  
        # shape: [batch_size, head, length, length]
        # → dot product 결과를 sqrt(d_tensor)로 나눠 안정화 (scale)

        # 2. 마스크 적용 (필요한 경우만)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)  
            # 마스킹된 위치는 매우 작은 값으로 만들어 softmax 후 0에 수렴하게 함

        # 3. softmax를 통해 attention weights로 변환 (확률 분포)
        score = self.softmax(score)  # shape: [batch_size, head, length, length]

        # 4. attention weights와 Value를 곱해 최종 출력 계산
        v = score @ v  # shape: [batch_size, head, length, d_tensor]

        return v, score  # 결과값과 attention map 반환
