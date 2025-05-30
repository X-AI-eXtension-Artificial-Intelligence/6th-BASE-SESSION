import torch.nn as nn
import torch

## 입력에 정규화하고 주어진 서브 레이어 실행 -> Dropout 적용하고 입력과 더해서 출력
class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, dr_rate=0, gated=True):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)
        self.gated = gated

        if gated:
            # Learnable gating scalar α, initialized at 0.5
            self.alpha = nn.Parameter(torch.tensor(0.5))  # shape: scalar

    def forward(self, x, sub_layer):
        residual = x
        out = self.norm(x)
        out = sub_layer(out)
        out = self.dropout(out)

        if self.gated:
            # α ∈ [0,1] using sigmoid
            gate = torch.sigmoid(self.alpha)
            out = gate * residual + (1 - gate) * out
        else:
            out = residual + out

        return out