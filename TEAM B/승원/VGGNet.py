import torch
import torch.nn as nn

# VGGNet 블록 정의
def conv_2_block(in_dim, out_dim):  # 2개의 컨볼루션 레이어를 쌓은 블록
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 레이어
        nn.ReLU(),  # 활성화 함수: ReLU 사용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 레이어
        nn.ReLU(),  # 활성화 함수
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링 레이어
    )

def conv_3_block(in_dim, out_dim):  # 3개의 컨볼루션 레이어를 쌓은 블록 (VGG 구조)
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 세 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링 (특징 맵 크기 절반으로 축소)
    )

# VGGNet 모델 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):  # base_dim: 기본 채널 크기, num_classes: 분류할 클래스 개수
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 첫 번째 컨볼루션 블록 (입력 3채널 → 출력 base_dim 채널)
            conv_2_block(base_dim, 2*base_dim),  # 두 번째 컨볼루션 블록 (출력 채널 2배 증가)
            conv_3_block(2*base_dim, 4*base_dim),  # 세 번째 컨볼루션 블록 (출력 채널 4배 증가)
            conv_3_block(4*base_dim, 8*base_dim),  # 네 번째 컨볼루션 블록 (출력 채널 8배 증가)
            conv_3_block(8*base_dim, 8*base_dim)  # 다섯 번째 컨볼루션 블록 (출력 채널 유지)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim, 4096),  # 완전 연결층: 512 → 4096
            nn.ReLU(True),  # 활성화 함수 (ReLU)
            nn.Dropout(),  # 드롭아웃 (과적합 방지)
            nn.Linear(4096, 1000),  # 완전 연결층: 4096 → 1000
            nn.ReLU(True),  # 활성화 함수 (ReLU)
            nn.Dropout(),  # 드롭아웃 (과적합 방지)
            nn.Linear(1000, num_classes)  # 출력층: 1000 → CIFAR-10 클래스 수 10으로 출력
        )

    def forward(self, x):  # 순전파 함수
        x = self.feature(x)  # 특징 추출 (Conv 블록 통과)
        x = x.view(x.size(0), -1)  # 2D 특징 맵을 1D 벡터로 변환
        x = self.fc_layer(x)  # 완전 연결층 통과 (분류 수행)
        return x