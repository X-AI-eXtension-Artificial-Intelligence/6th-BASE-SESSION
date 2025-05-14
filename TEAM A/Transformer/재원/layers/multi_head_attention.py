from torch import nn


# 기본 틀인 ScaledDotProductAttention으로 나머지 multihead attention 구성
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head # 헤드 수
        self.attention = ScaleDotProductAttention() #어텐션
        self.w_q = nn.Linear(d_model, d_model) #query 생성하는 행렬
        self.w_k = nn.Linear(d_model, d_model) #key 생성 행렬
        self.w_v = nn.Linear(d_model, d_model) #value 생성 행렬
        self.w_concat = nn.Linear(d_model, d_model) #세개 concat하는 행렬

    def forward(self, q, k, v, mask=None): 
        # 선형 변환 층으로 q,k,v 생성
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # head 만큼 분할
        q, k, v = self.split(q), self.split(k), self.split(v)

        # attention 수행
        out, attention = self.attention(q, k, v, mask=mask)

        # 헤드 결과 결합 + 선형 변환
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    # 입력 텐서 분할하는 함수
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head #헤드 수에 따른 차원 크기
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) #d_model을 헤드 * 각 헤드 차원으로 쪼갠다음에 헤드를 앞으로 보내서 독립적으로 헤드별 연산 수행
  
        return tensor

    def concat(self, tensor):
 
        #위에서 변환된 텐서의 형태
        batch_size, head, length, d_tensor = tensor.size()
        #모델 차원 역변환
        d_model = head * d_tensor

        #차원 다시 변환(transpose 뒤에 바로 view 사용 불가 - 메모리 연속성 보장) 다시 원래대로
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
