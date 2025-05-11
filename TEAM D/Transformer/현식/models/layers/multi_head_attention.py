from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    # d_model : 모델 차원 (ex : 512)
    # n_head : attention head 개수 (ex : 8)
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        # ScaleDotProductAttention 클래스를 이용해 attention score 계산
        self.attention = ScaleDotProductAttention()
        # Query, Key , Value를 만들기 위한 Linear 변    환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 여러 개의 attention haed를 합친 후 최종 출력을 만들기 위한 Linear 변환
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # Query, Key, Value 변환
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Multi-Hade Attention을 위한 분할
        q, k, v = self.split(q), self.split(k), self.split(v)

        # Scale Dot Product Attention 수행
        out, attention = self.attention(q, k, v, mask=mask)

        # 여러 head를 다시 결합
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
