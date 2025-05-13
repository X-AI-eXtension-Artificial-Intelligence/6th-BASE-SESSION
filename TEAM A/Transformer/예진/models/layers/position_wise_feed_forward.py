"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    위치별 완전연결 신경망 (Position-wise Feed Forward Network)

    Transformer에서 각 단어 벡터(위치)에 대해 독립적으로 적용되는 MLP
    구조: Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        d_model: 입력 및 출력 벡터 차원
        hidden: 중간 은닉층 차원 (확장된 차원)
        drop_prob: Dropout 비율 (과적합 방지용)
        """
        super(PositionwiseFeedForward, self).__init__()

        # 선형 계층1: 입력 벡터 더 큰 차원으로 확장
        self.linear1 = nn.Linear(d_model, hidden)

        # 선형 계층2: 다시 원래 차원으로 축소
        self.linear2 = nn.Linear(hidden, d_model)

        # 비선형 활성화 함수: 표현력 향상
        self.relu = nn.ReLU()

        # 중간층 dropout: 일부 뉴런 무작위 제거 -> 일반화 성능 증가
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        x: 입력 텐서 (shape: [batch_size, seq_len, d_model])
        각 단어(토큰) 벡터에 대해 동일한 방식으로 독립 처리
        """
        x = self.linear1(x)   # 선형 확장: [batch, seq_len, hidden]
        x = self.relu(x)      # ReLU 적용: 비선형성 도입
        x = self.dropout(x)   # 과적합 방지를 위한 dropout
        x = self.linear2(x)   # 다시 축소하여 d_model 차원으로 복원
        return x              # 최종 출력: [batch, seq_len, d_model]
