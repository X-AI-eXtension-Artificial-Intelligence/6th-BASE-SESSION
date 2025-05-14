from torch import nn


class PositionwiseFeedForward(nn.Module): # 위치별 피드포워드 네트워크 클래스 정의

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 첫 번째 선형 변환 레이어 (입력 차원 -> 은닉 차원)
        self.linear2 = nn.Linear(hidden, d_model) # 두 번째 선형 변환 레이어 (은닉 차원 -> 입력 차원)
        self.relu = nn.ReLU() # 활성화 함수 객체 생성
        self.dropout = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
