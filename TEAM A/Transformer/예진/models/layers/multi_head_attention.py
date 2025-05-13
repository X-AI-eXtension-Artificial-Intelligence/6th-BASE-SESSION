from torch import nn
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 모듈
    - 입력을 여러 개의 head로 나눠 병렬로 attention 연산 수행
    - 각 head는 서로 다른 attention 수행
    - 결과 다시 합쳐서 최종 출력 생성
    """

    def __init__(self, d_model, n_head):
        """
        d_model: 전체 벡터 차원
        n_head: head의 개수
        """
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head  # head 수 저장

        # 실제 scaled dot-product attention을 계산할 모듈
        self.attention = ScaleDotProductAttention()

        # 입력 벡터를 각각 Query, Key, Value로 변환하는 선형 레이어
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 여러 head의 출력을 다시 하나로 합친 뒤 선형 변환
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: Query, Key, Value 입력 (보통 shape = [batch, seq_len, d_model])
        mask: attention 연산 시 특정 위치를 무시하기 위함
        """

        # 1. Q, K, V 각각 선형 변환
        # 입력 벡터를 d_model 차원으로 투영
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 여러 head로 분할
        # [batch, seq_len, d_model] -> [batch, n_head, seq_len, d_model // n_head]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 각 head별로 Scaled Dot-Product Attention 수행
        # head 수 만큼 attention 연산을 병렬로 실행
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. head들의 출력을 다시 하나로 합침
        # [batch, n_head, seq_len, d_head] -> [batch, seq_len, d_model]
        out = self.concat(out)

        # 5. 선형 변환 (다음 층으로 넘길 수 있게)
        out = self.w_concat(out)

        return out  

    def split(self, tensor):
        """
        입력 벡터를 n_head 개수만큼 분할

        입력: [batch_size, seq_len, d_model]
        출력: [batch_size, n_head, seq_len, d_model // n_head]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head  # 각 head의 차원 수

        # head 차원을 새로 추가하고, shape을 [batch, n_head, length, d_tensor]로 변경
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor)
        tensor = tensor.transpose(1, 2)  # head 축을 앞으로

        return tensor

    def concat(self, tensor):
        """
        여러 head의 출력을 다시 하나로 합침

        입력: [batch_size, n_head, seq_len, d_tensor]
        출력: [batch_size, seq_len, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor  # 원래 차원으로 복원

        # head 축을 뒤로 보내고 이어붙임: [batch, seq_len, d_model]
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor
