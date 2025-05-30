"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn
import torch.nn.functional as F

class CosineSimilarityAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, temperature = 0.1):
        super(CosineSimilarityAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = temperature

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        q_norm = F.normalize(q, dim=-1, eps=e)
        k_norm = F.normalize(k, dim=-1, eps=e)
        k_t_norm = k_norm.transpose(2, 3)  # transpose

        score = q_norm @ k_t_norm
        score = score / self.temperature  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
