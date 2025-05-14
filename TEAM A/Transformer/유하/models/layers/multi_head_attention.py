from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head 
        self.attention = ScaleDotProductAttention() # ScaleDotProductAttention 객체 생성
        self.w_q = nn.Linear(d_model, d_model) # query용 선형 변환 레이어
        self.w_k = nn.Linear(d_model, d_model) # key용 선현 변환 레이어
        self.w_v = nn.Linear(d_model, d_model) # value용 선형 변환 레이어
        self.w_concat = nn.Linear(d_model, d_model) # 여러 헤드의 출력을 합친 뒤, 다시 임베딩 차원으로 변환하는 레이어 생성

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # 각각 선형 변환 적용

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v) # 각 텐서를 헤드 수만큼 분할

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask) # 각 헤드별로 ScaleDotProductAttention 수행

        # 4. concat and pass to linear layer
        out = self.concat(out) # 여러 헤드 출력 합침
        out = self.w_concat(out) # 합친 출력에 선형 변환 적용

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor): # 입력 텐서를 헤드 수만큼 분할해 [batch, head, length, d_tensor] 형태로 변환
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor): # 여러 헤드로 분할된 텐서를 다시 [batch, length, d_model] 형태로 합침
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
