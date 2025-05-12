import math

from torch import nn

# 스케일 닷 어텐션의 구현 (Query : 디코더, Key : 인코더, Value : 인코더)
class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # 마지막 차원에서 소프트맥스 수행(어텐션 Score)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 4차원 텐서(배치 사이즈, 헤드 수, 시퀀스 길이, 한 개 헤드의 벡터 차원)
        batch_size, head, length, d_tensor = k.size()

        # 내적을 위해서 전치하고 루트 dk로 나누어 scaled dot 내적한 score 계산
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 마스킹 적용 (필요하면)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000) #마스크 0인 부분에 -10000 아주 작은 값값

        # 소프트 맥스에 태워서 실제 score 계산
        score = self.softmax(score)

        # Value 업데이트
        v = score @ v

        return v, score # 어텐션 결과로 나온 vector와 그때의 attention value 계산
