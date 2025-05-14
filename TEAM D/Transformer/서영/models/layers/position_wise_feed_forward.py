"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):
    # Transformer의 각 인코더/디코더 레이어 내부에서 사용하는
    # 포지션 독립적인 피드포워드 네트워크 정의

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # 첫 번째 선형 변환: d_model → hidden
        self.linear1 = nn.Linear(d_model, hidden)

        # 두 번째 선형 변환: hidden → d_model
        self.linear2 = nn.Linear(hidden, d_model)

        # 비선형 활성화 함수 (ReLU)
        self.relu = nn.ReLU()

        # 드롭아웃으로 과적합 방지
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 입력 x는 [batch_size, seq_len, d_model] 형상

        x = self.linear1(x)   # 선형 변환 1
        x = self.relu(x)      # ReLU 비선형성 추가
        x = self.dropout(x)   # 드롭아웃 적용
        x = self.linear2(x)   # 선형 변환 2

        return x              # shape 동일: [batch_size, seq_len, d_model]
