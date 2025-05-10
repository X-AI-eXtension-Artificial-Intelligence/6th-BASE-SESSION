
import math
from torch import nn 

class ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention 구현 클래스

    - Query: 현재 집중하고 있는 단어 벡터
    - Key: 각 단어가 담고 있는 정보의 '주소'
    - Value: 실제 단어 정보
    - 어텐션은 Q·K^T(트랜스포즈)를 통해 유사도를 측정하고, 이를 V에 곱해 결과 생성
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)  # 마지막 차원 기준 softmax 적용 (유사도 정규화용)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        :param q: Query 텐서 [batch_size, head, length_q, d_tensor]
        :param k: Key 텐서   [batch_size, head, length_k, d_tensor]
        :param v: Value 텐서 [batch_size, head, length_v, d_tensor]
        :param mask: 어텐션 마스킹 (optional: pad or look-ahead mask)
        :param e: 작은 수, 나눗셈 안정성 보장을 위한 epsilon
        """

        # 입력 텐서 크기 확인 (Key 기준)
        batch_size, head, length, d_tensor = k.size()

        # Key를 전치 Q와 내적 가능하게게 (K: [B, H, L, D] → [B, H, D, L])
        k_t = k.transpose(2, 3)

        # 2. Q와 K^T의 행렬곱을 통해 유사도(score) 계산 후, sqrt(d_tensor)로 나눠 스케일 조정
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 3. 마스크가 있다면, 가려야 할 위치(score==0)에 매우 작은 값(-10000) 넣어 softmax 결과 거의 0 되게 함
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 4. softmax로 유사도를 0~1 사이로 정규화 (attention weight 생성)
        score = self.softmax(score)

        # 5. attention weight(score)를 Value에 곱해서 context vector 생성
        v = score @ v  # 결과 shape: [batch_size, head, length_q, d_tensor]

        return v, score  
