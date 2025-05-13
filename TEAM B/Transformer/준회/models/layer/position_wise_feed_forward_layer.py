import torch.nn as nn

## FFNN : FC Layer 두 번
class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2, dr_rate=0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1  ## 첫 번째 FC 레이어: 입력 차원(d_embed) → 확장 차원(d_ff)
        self.relu = nn.ReLU()  ## 활성화 함수 (비선형 변환)
        self.dropout = nn.Dropout(p=dr_rate)  ## 드롭아웃 (과적합 방지)
        self.fc2 = fc2  ## 두 번째 FC 레이어: 축소된 차원(d_ff) → 원래 차원(d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)     ## 첫 번째 완전 연결 층 (d_embed → d_ff)
        out = self.relu(out)    ## 비선형 활성화 함수 적용 (ReLU)
        out = self.dropout(out) ## 드롭아웃 적용 (과적합 방지)
        out = self.fc2(out)     ## 두 번째 완전 연결 층 (d_ff → d_embed)
        return out              ## 최종 출력 반환