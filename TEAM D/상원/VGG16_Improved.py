#기존 VGG16 대비 개선된 점 요약

# LeakyReLU(0.1) 적용 → 기울기 소실 문제 해결
# Batch Normalization 추가 → 학습 안정성 향상 및 빠른 수렴
# Global Average Pooling(GAP) 적용 → 연산량 감소 및 일반화 성능 향상
# Fully Connected Layer 축소 → 4096 → 512 뉴런으로 파라미터 수 감소
# Softmax 제거 → CrossEntropyLoss에서 자동 적용됨


import torch.nn as nn
import torch.nn.functional as F

# 두 개의 Conv → BatchNorm → LeakyReLU + MaxPool 블록

def conv_2_block(in_dim, out_dim):
  # 2D 컨볼루션(Convolutional) 연산을 수행하는 PyTorch의 클래스
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1, inplace=True),
        # MaxPooling을 통해 공간 정보 압축 및 연산량 감소
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

# 세 개의 Conv → BatchNorm → LeakyReLU + MaxPool 블록
# 더욱 깊은 특징 추출
def conv_3_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1, inplace=True),

        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1, inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2)
    )

# 개선된 VGG16 모델
class VGG16_Improved(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(VGG16_Improved, self).__init__()

        self.features = nn.Sequential(
            conv_2_block(3, base_dim),               # 64
            conv_2_block(base_dim, 2*base_dim),      # 128
            conv_3_block(2*base_dim, 4*base_dim),    # 256
            conv_3_block(4*base_dim, 8*base_dim),    # 512
            conv_3_block(8*base_dim, 8*base_dim),    # 512
        )

        # Global Average Pooling (GAP)
        # 출력 피처맵의 크기를 1x1로 줄여서 파라미터 수를 감소시키고, 일반화 성능 향상
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 

        self.classifier = nn.Sequential(
          
            nn.Flatten(), # GAP의 출력을 1차원 벡터로 변환
            nn.Linear(8*base_dim, 512), # Fully Connected Layer (512 노드)
            nn.LeakyReLU(0.1, inplace=True),# 활성화 함수 적용
            nn.Dropout(0.5), # 과적합 방지를 위한 드롭아웃 (50%)
            nn.Linear(512, num_classes)              
            # softmax는 사용하지 않음
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x  # softmax는 CrossEntropyLoss에서 자동 적용
  # softmax를 출력층에서 사용하지 않는 이유는 PyTorch의 CrossEntropyLoss 함수가 내부적으로 softmax를 적용하기 때문입니다.
