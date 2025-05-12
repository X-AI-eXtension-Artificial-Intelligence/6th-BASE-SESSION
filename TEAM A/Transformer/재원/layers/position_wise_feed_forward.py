from torch import nn

# FeedForward 각 토큰별로 독립적으로 처리 h 차원에서 hidden 으로 변환했다가 다시 h 차원으로
class PositionwiseFeedForward(nn.Module):

    # Attention은 단어 토큰 간 연관성이고, hidden layer 둬서 단어 개별 학습
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
