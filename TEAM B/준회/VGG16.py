## 모델 정의 코드

'''

- 3x3 합성곱 연산 x2 (채널 64)
- 3x3 합성곱 연산 x2 (채널 128)
- 3x3 합성곱 연산 x3 (채널 256)
- 3x3 합성곱 연산 x3 (채널 512)
- 3x3 합성곱 연산 x3 (채널 512)
- FC layer x3
  - FC layer 4096
    - FC layer 4096
    - FC layer 1000
    
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

## nn.Conv2d는 2차원 입력을 위한 Conv 레이어, 2차원 데이터에 여러 필터를 적용해 특징을 추출하는 역할
## in_dim은 입력 채널의 수, 예를 들어 흑백 이미지는 1, RGB 컬러 이미지는 3
## out_dim은 출력 채널의 수, 필터의 수, 모델이 얼마나 많은 특징을 추출할 지 결정
## kernel_size = 3은 필터의 크기를 3 x 3으로 설정
## padding = 1은 입력 데이터 주변을 0으로 채워 출력 데이터의 크기가 입력 데이터의 크기와 동일하게 유지
## nn.MaxPool2d는 feature map의 크기를 줄이는 데 사용, 2 x 2 크기의 윈도우로 2칸씩 이동하며 적용

## conv 블럭이 2개인 경우 (conv + conv + max pooling)
def conv2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.GELU(),
        nn.MaxPool2d(2, 2)
    )
    return model

## conv 블럭이 3개인 경우 (conv + conv + conv + max pooling)
def conv3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(), 
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        nn.GELU(),
        nn.MaxPool2d(2, 2)
    )
    return model

## define VGG16
class VGG16(nn.Module):

    ## 모델 예측에 사용할 데이터 CIFAR 10의 클래스 개수가 10
    def __init__(self, base_dim, num_classes = 10):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv2_block(3, base_dim), ## RGB -> 64채널
            conv2_block(base_dim, base_dim), ## 64 -> 128
            conv3_block(base_dim, 4 * base_dim), ## 128 -> 256
            conv3_block(4 * base_dim, 8 * base_dim), ## 256 -> 512
            conv3_block(8 * base_dim, 8 * base_dim) ## 512 -> 512
        )

        self.fc_layer = nn.Sequential(
            ## CIFAR10은 크기가 32 X 32 이므로
            nn.Linear(8 * base_dim * 7 * 7, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 2000), ## 8192 -> 2000
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, num_classes) ## 2000 -> 10
        )
    
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x