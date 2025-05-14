"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    Transformer Position-wise Feed Forward Network
    - 각 position(단어 벡터)마다 동일한 FeedForward Network 적용
    - 즉, 문장의 각 단어 벡터에 독립적으로 MLP 적용
    """

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        초기화 함수

        :param d_model: 입력/출력 벡터 차원 (예: 512)
        :param hidden: FeedForward 은닉층 차원 (예: 2048)
        :param drop_prob: dropout 확률
        """
        super(PositionwiseFeedForward, self).__init__()

        # 첫 번째 Linear layer: (d_model → hidden)
        self.linear1 = nn.Linear(d_model, hidden)

        # 두 번째 Linear layer: (hidden → d_model)
        self.linear2 = nn.Linear(hidden, d_model)
        
        # 활성화 함수: ReLU
        self.relu = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        forward 함수
        - 입력 x에 feedforward 연산 적용

        :param x: 입력 벡터 (batch_size x seq_len x d_model)
        :return: 출력 벡터 (batch_size x seq_len x d_model)
        """

        x = self.linear1(x)   # 첫 번째 Linear: [batch, seq_len, d_model] → [batch, seq_len, hidden]
        x = self.relu(x)      # ReLU 비선형성
        x = self.dropout(x)   # Dropout
        x = self.linear2(x)   # 두 번째 Linear: [batch, seq_len, hidden] → [batch, seq_len, d_model]
        return x