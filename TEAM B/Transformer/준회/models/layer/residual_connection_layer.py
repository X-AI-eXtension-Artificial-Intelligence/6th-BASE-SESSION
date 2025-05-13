import torch.nn as nn

## 입력에 정규화하고 주어진 서브 레이어 실행 -> Dropout 적용하고 입력과 더해서 출력
class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)

    ## sub_layer는 attention 연산이나 FFNN을 의미
    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out