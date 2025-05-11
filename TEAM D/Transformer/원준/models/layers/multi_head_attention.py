
from torch import nn  

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    멀티헤드 어텐션 구현 클래스
    입력을 여러 개의 어텐션 헤드로 나눈 뒤 각각 어텐션 수행 후 결합
    각자 역할 분담하는 것 
    """

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head  # 어텐션 헤드 개수   논문에서는 8개 사용 
        self.attention = ScaleDotProductAttention()  # 스케일드 닷 프로덕트 어텐션 모듈

        # 입력을 query, key, value로 변환하기 위한 선형 변환 레이어
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 여러 헤드의 출력을 합친 뒤 다시 d_model 차원으로 만드는 선형 레이어
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. 입력 q, k, v에 대해 선형 변환 수행
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 멀티헤드 구조를 위해 각 텐서를 헤드 수만큼 나누기
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 각 헤드에 대해 스케일드 닷 프로덕트 어텐션 수행
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 여러 헤드의 결과를 다시 하나로 결합(concat)하고 선형 변환
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. attention map 시각화는 향후 구현 예정
        return out

    def split(self, tensor):
        """
        입력 텐서를 헤드 수(n_head)만큼 분할하여 헤드 단위로 어텐션을 수행할 수 있도록 변환

        tensor: [batch_size, length, d_model]
        :return: [batch_size, n_head, length, d_tensor]로 변경
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head  # 각 헤드의 차원 크기     
        # d_model=512, n_head=8 → d_tensor=64

        # 뷰(view) 및 전치(transpose)로 차원을 재배치
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # 어텐션 헤드 중심 구조로 바뀐 것   [batch_size, n_head, length, d_tensor]

        return tensor

    def concat(self, tensor):
        """
        split()에서 나눈 여러 헤드를 다시 하나의 d_model 차원으로 합치는 함수

        :param tensor: [batch_size, n_head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor  # 원래 차원 복원

        # 전치 및 view로 다시 [batch_size, length, d_model] 형태로 결합
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
