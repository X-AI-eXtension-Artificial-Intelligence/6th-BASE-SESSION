import math

from torch import nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        # softmax 함수를 사용하여 attention 가중치를 확률값(0~1)으로 변환
        self.softmax = nn.Softmax(dim=-1)

    # 입력(q, k, v): Query, Key, Value (batch_size, n_head, seq_len, d_tensor)
    def forward(self, q, k, v, mask=None, e=1e-12):
        # Key의 크기를 가져옴
        batch_size, head, length, d_tensor = k.size()

        # Query와 Key의 내적 (유사도 계산)
        k_t = k.transpose(2, 3)   
        score = (q @ k_t) / math.sqrt(d_tensor)  

        # masking 적용 (옵선)
        # Encoder에서는 적용 X
        # Decoder에서는 적용 O
        if mask is not None:
            # mask ==0인 부분을 -1000으로 채움
            # softmax를 거지면 -1000은 거의 0에 가까운 확률로 변환됨 => 무시됨
            score = score.masked_fill(mask == 0, -10000)
 
        # softmzx를 적용하여 확률값으로 변환
        score = self.softmax(score)

        # value에 가중치 적용
        v = score @ v

        return v, score
