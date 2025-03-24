import torch
import torch.nn as nn
from tqdm import trange  # 학습 진행 과정 시각적으로 보여주는 라이브러리

# 학습률 설정
learning_rate = 0.001

# 두 개의 컨볼루션 레이어를 포함하는 블록 정의
def conv_2_block(in_dim, out_dim):  # in_dim: 입력 이미지 또는 특징 맵의 채널 수 out_dim: 출력 특징 맵의 채널 수
    model = nn.Sequential(         #  nn.Sequential은 파이토치에서 여러 층을 순차적으로 쌓을 때 사용하는 컨테이너
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 컨볼루션 레이어
        nn.ReLU(), 
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 컨볼루션 레이어
        nn.ReLU(), 
        nn.MaxPool2d(2, 2)  # kernel_size=2 * 2 , stride=2
    )
    return model

# 세 개의 컨볼루션 레이어를 포함하는 블록 정의
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  
    )
    return model

# VGG 모델 정의
class VGG(nn.Module): # nn.Module은 PyTorch에서 모든 신경망 모델의 기본 클래스
    def __init__(self, base_dim, num_classes=10):  # 생성자 자동 실행, base_dim 모델 기본 채널 크기 설정, 분류할 클래스 개수
        super(VGG, self).__init__()   # nn.Module의 기능을 상속받기 위해 super() 호출 
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 입력 채널 3(RGB), 출력 채널 base_dim,  위에서 정의한 것 
            conv_2_block(base_dim, 2 * base_dim),  # 채널 수 2배로 증가
            conv_3_block(2 * base_dim, 4 * base_dim),
            conv_3_block(4 * base_dim, 8 * base_dim),
            conv_3_block(8 * base_dim, 8 * base_dim),
        )
        self.fc_layer = nn.Sequential(  # 완전 연결 계층 
            nn.Linear(8 * base_dim * 1 * 1, 4096),  # CIFAR-10 입력 크기에 맞게 설정
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000), # 4096차원 벡터 입력받아 1000차원 벡터로 변환
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),  # 클래스 예측 
        )

    def forward(self, x): # 모델이 데이터를 처리하는 방법을 정의하는 함수
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # view 크기 변경 
                                    # x.size(0) 배치 사이즈 
                                    # 함수텐서를 펼쳐서 Fully Connected Layer에 입력 일렬로 배열열
        x = self.fc_layer(x)
        return x

