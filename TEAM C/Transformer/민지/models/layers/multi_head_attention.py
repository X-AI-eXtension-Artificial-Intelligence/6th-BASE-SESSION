import torch
import torch.nn as nn
import math
### scaled dot product attention을 포함한 것이 MHSA임
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, drop_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)  # 512x512
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, q, k, v, mask=None):
        # 1. 선형변환 수행하고 -> (Tx512) * (512x512) ==> (Tx512)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 각각 Tx64 차원으로 분할
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. scale dot product 수행 (similarity 계산)
        out, attention = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # 4. concat and linear
        out = self.concat(out)

        out = self.dropout(out)

        return out

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # input : 4 dimension tensor
        batch_size, n_head, length, d_qkv = k.size()

        k_t = k.transpose(2, 3)  # transpose 시키고
        score = (q @ k_t) / math.sqrt(d_qkv)  # scaled dot product

        # decoder에서
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)  # mask 텐서에서 값이 0인 위치에 -10000을 할당하는 역할

        softmax = nn.Softmax(dim=-1)
        score = softmax(score)

        v = score @ v
        # 차원은 변함없이 Tx64임
        return v, score

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_qkv = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_qkv).transpose(1, 2)
        #                                         기존 512 ==> 8x64로 변경하고, 트랜스포즈 length랑 head 차원 변경
        return tensor

    def concat(self, tensor):
        """
        총 8개의 head에서 구해진 attention 값들을 concatnate하고 -> 리니어로 보냄
        :param tensor: [batch_size, head, length, d_qkv]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_qkv = tensor.size()
        d_model = head * d_qkv # 8x64=512 다시 합침

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        # [batch_size, head, length, d_qkv] -> [batch_size, length, head, d_qkv]
        # transpose 연산 후 텐서는 메모리 상에서 비연속적인 배열일 수 있음 메모리 상에서 연속되지 않은 경우 연산에 문제가 생길 수 있으므로
        # contiguous()를 호출하여 텐서를 연속적인 메모리 블록으로 변환함
        # 그리고 다시 뒤의 두 차원을 d_model로 합침

        w_concat = nn.Linear(d_model, d_model)
        tensor = w_concat(tensor)

        return tensor