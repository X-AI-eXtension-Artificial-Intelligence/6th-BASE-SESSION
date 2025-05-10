from torch import nn 

class PositionwiseFeedForward(nn.Module):
    """
    Transformer에서 사용되는 Position-wise Feed Forward Network 구현
    각 단어 벡터 위치별로 독립적으로 동일한 FFN을 적용함
    문장 안 단어 벡터마다
    """

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
         d_model: 입력 및 출력 차원 (Transformer 모델 차원)
         hidden: 내부 은닉층의 차원 (일반적으로 d_model보다 큼)
         drop_prob: 드롭아웃 확률
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)      # 첫 번째 선형 변환: d_model → hidden
        self.linear2 = nn.Linear(hidden, d_model)      # 두 번째 선형 변환: hidden → d_model
        self.relu = nn.ReLU()                          # 비선형 활성화 함수
        self.dropout = nn.Dropout(p=drop_prob)         # 드롭아웃 
        
    def forward(self, x):
        x = self.linear1(x)      # 첫 번째 선형 레이어 통과
        x = self.relu(x)         # ReLU 활성화 함수 적용
        x = self.dropout(x)      # 드롭아웃 적용
        x = self.linear2(x)      # 두 번째 선형 레이어 통과
        return x                 # 최종 출력 반환
