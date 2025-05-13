import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)