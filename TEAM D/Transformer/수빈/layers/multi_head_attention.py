"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
'''
MultiHeadAttention 클래스는 “여러 개의 attention을 병렬로 계산하고 합쳐주는 클래스”

- Query, Key, Value를 각각 선형변환 (projection)
- 입력 벡터를 head 개수만큼 분리
- 각 head에 대해 attention 계산
- 각 head 결과를 합치고 projection
'''
'''
입력 (q, k, v)
 └→ Linear projection (w_q, w_k, w_v)
     └→ split (head로 분리)
         └→ 각 head에 Scaled Dot-Product Attention
             └→ concat (다시 합치기)
                 └→ 최종 projection (w_concat)
                     └→ 출력
'''
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Transformer Multi-Head Attention 클래스
    - 입력 벡터를 여러 head로 분리해 각각 attention 계산
    - 각 head의 결과를 합쳐 최종 attention 출력
    """

    def __init__(self, d_model, n_head):
        """
        초기화 함수
        :param d_model: 임베딩 차원
        :param n_head: attention head 개수
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head                        # head 개수 저장
        self.attention = ScaleDotProductAttention() # head별 attention 계산 모듈

        # query, key, value를 각각 linear projection하는 layer
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection

        self.w_concat = nn.Linear(d_model, d_model) # 여러 head 결과를 합친 뒤 projection

    def forward(self, q, k, v, mask=None):
        """
        forward 함수
        - 입력 q, k, v에 multi-head attention 계산

        :param q: Query 벡터 (batch_size x seq_len x d_model)
        :param k: Key 벡터 (batch_size x seq_len x d_model)
        :param v: Value 벡터 (batch_size x seq_len x d_model)
        :param mask: attention mask
        :return: output (batch_size x seq_len x d_model)
        """

        # 1. 입력 q, k, v에 projection
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # shape: (batch_size, seq_len, d_model)

        # 2. head 개수만큼 q, k, v 분리
        q, k, v = self.split(q), self.split(k), self.split(v)
        # shape: (batch_size, n_head, seq_len, d_tensor)

        # 3. 각 head에 대해 scaled dot-product attention 계산
        out, attention = self.attention(q, k, v, mask=mask)
        # out: (batch_size, n_head, seq_len, d_tensor)

        # 4. head별 결과를 다시 하나로 concat
        out = self.concat(out)      # shape: (batch_size, seq_len, d_model)

        # 5. 최종 projection
        out = self.w_concat(out)    # shape: (batch_size, seq_len, d_model)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        입력 벡터를 head 개수로 분리하는 함수

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head   # head당 차원

        # (batch_size, seq_len, n_head, d_tensor) → (batch_size, n_head, seq_len, d_tensor)
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        분리된 head 벡터들을 다시 하나로 합치는 함수

        :param tensor: [batch_size, n_head, seq_len, d_tensor]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor    # 원래 차원 복원

        # (batch_size, n_head, seq_len, d_tensor) → (batch_size, seq_len, n_head, d_tensor)
        # → (batch_size, seq_len, d_model)
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
