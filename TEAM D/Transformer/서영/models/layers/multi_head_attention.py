"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

# Scaled Dot-Product Attention 모듈 불러오기
from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    # 다중 헤드 어텐션 클래스 정의 (Transformer 핵심 구성 요소)

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head  # 헤드 개수 저장
        self.attention = ScaleDotProductAttention()  # scaled dot-product attention 모듈

        # 쿼리, 키, 밸류를 위한 선형 변환층 (d_model → d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 여러 헤드의 결과를 다시 하나로 합치는 선형 변환층
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. 입력 q, k, v에 대해 각각 선형 변환 수행
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 각 텐서를 n_head 개수만큼 분할 (멀티헤드 구성)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 각 헤드에 대해 Scaled Dot-Product Attention 수행
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 헤드들을 concat하고 선형 변환 수행
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. Attention 시각화(예정) - 구현 예정
        # TODO : we should implement visualization

        return out  # shape: [batch_size, seq_len, d_model]

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head  # 각 헤드별 차원 크기

        # [B, L, D] → [B, L, H, D/H] → [B, H, L, D/H]
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        # 결과적으로 각 헤드별로 독립된 어텐션 계산이 가능해짐
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor  # 원래 차원 복원

        # [B, H, L, D/H] → [B, L, H, D/H] → [B, L, D]
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor
