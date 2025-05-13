import math
import torch
from torch import nn

# layer_norm
class LayerNorm(nn.Module):
    # 레이어 정규화(Layer Normalization) 구현 클래스.
    # 입력 텐서의 마지막 차원(특성 차원)을 따라 평균과 분산을 구해 정규화함.
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # 정규화된 값에 곱할 scale(gamma)과 shift(beta) 파라미터 (학습됨)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 마지막 차원(-1, 즉 d_model 차원) 기준으로 평균과 분산 계산
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1'은 마지막 차원을 의미

        # 정규화: (입력 - 평균) / (표준편차)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta # scale(gamma) 곱하고, shift(beta) 더함
        return out
    
# mult_head_attention
class MultiHeadAttention(nn.Module):
    # 입력을 여러 헤드로 쪼개 병렬로 어텐션을 수행 후 합침.
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()

        # 쿼리/키/밸류 생성용 선형 변환 레이어
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model) # 여러 헤드 결과를 합쳐 다시 d_model로 변환

    def forward(self, q, k, v, mask=None):
        # 1. 입력을 쿼리/키/밸류로 선형 변환
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 각 헤드로 분할 (shape: [batch, head, length, d_tensor])
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 각 헤드별로 스케일 닷 프로덕트 어텐션 수행
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 여러 헤드의 출력을 합치고 선형 변환
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. 어텐션 맵 시각화 (TODO)
        # TODO : 시각화 기능 구현 필요

        return out

    def split(self, tensor):
        # 입력 텐서를 헤드 개수만큼 분할
        # tensor: [batch_size, length, d_model]
        # return: [batch_size, head, length, d_tensor]

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head # 각 헤드별 차원
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # (batch, length, n_head, d_tensor)로 reshape 후, (batch, n_head, length, d_tensor)로 transpose
        # 컨볼루션 신경망(CNN)에서 그룹 컨볼루션을 적용할 때 채널을 여러 그룹으로 나누는 방식과 유사

        return tensor

    def concat(self, tensor):
        # split의 역함수: 여러 헤드의 출력을 다시 합침
        # tensor: [batch_size, head, length, d_tensor]
        # return: [batch_size, length, d_model]
        
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        # (batch, length, head, d_tensor)로 transpose 후, (batch, length, d_model)로 reshape

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

# position_wise_feed_forward
class PositionwiseFeedForward(nn.Module):
    # 포지션별 피드포워드 네트워크.
    # 각 위치별로 독립적으로 2개의 선형 레이어와 ReLU, 드롭아웃을 적용.
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 첫 번째 선형 변환: d_model -> hidden
        self.linear1 = nn.Linear(d_model, hidden)
        # 두 번째 선형 변환: hidden -> d_model
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. 첫 번째 선형 변환 + ReLU
        x = self.linear1(x)
        x = self.relu(x)
        # 2. 드롭아웃 적용
        x = self.dropout(x)
        # 3. 두 번째 선형 변환
        x = self.linear2(x)
        return x
    
# scale_dot_product_attention
class ScaleDotProductAttention(nn.Module):
    # 스케일 닷 프로덕트 어텐션 구현 클래스.
    # 쿼리(Q), 키(K), 밸류(V)를 받아 어텐션 맵과 출력을 계산함.
    # Query : 현재 집중하는 문장(디코더)
    # Key : Query와의 관계를 확인할 모든 문장(인코더)
    # Value : Key와 동일한 모든 문장(인코더)

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # 어텐션 스코어를 확률로 변환

    def forward(self, q, k, v, mask=None, e=1e-12):
        # 입력 4차원 텐서 shape: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. 쿼리와 키의 내적을 통해 유사도(어텐션 스코어) 계산
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaling 하고 dot product

        # 2. 마스킹 적용 (옵션, 예: 패딩 마스킹, 미래 정보 차단 등)
        if mask is not None: 
            score = score.masked_fill(mask == 0, -10000) # 마스킹된 위치는 매우 작은 값으로 설정

        # 3. 소프트맥스 적용하여 확률 분포로 변환
        score = self.softmax(score)

        # 4. 어텐션 가중치(score)와 밸류(v)를 곱해 최종 출력 계산
        v = score @ v

        return v, score # (어텐션 적용 결과, 어텐션 맵)