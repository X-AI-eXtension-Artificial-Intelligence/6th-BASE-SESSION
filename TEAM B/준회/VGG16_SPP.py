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

## SPP 레이어 정의
class SPPLayer(nn.Module):
    def __init__(self, levels):
        super(SPPLayer, self).__init__()
        self.levels = levels  ## 풀링 크기 수준(예: 1x1, 2x2, 4x4)

    def forward(self, x):
        bs, c, h, w = x.size()  ## 배치크기, 채널, 높이, 너비
        spp_out = []

        for level in self.levels:
            kernel_size = (h // level, w // level)
            stride = (h // level, w // level)

            ## 각 level에 맞는 출력 크기로 adaptive pooling 적용
            pooling = F.adaptive_max_pool2d(x, output_size=(level, level))

            ## 결과를 1차원으로 펼쳐서 리스트에 추가
            spp_out.append(pooling.view(bs, -1))
            
        ## 모든 level의 풀링 결과를 이어붙임
        return torch.cat(spp_out, dim=1)
        
## VGG16 모델에 SPPNet 구조를 결합
class VGG16_SPP(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG16_SPP, self).__init__()
        self.feature = nn.Sequential(
            conv2_block(3, base_dim),
            conv2_block(base_dim, base_dim),
            conv3_block(base_dim, 4 * base_dim),
            conv3_block(4 * base_dim, 8 * base_dim),
            conv3_block(8 * base_dim, 8 * base_dim),
        )

        self.spp = SPPLayer(levels=[1, 2, 4])  # 다양한 크기의 pooling 수행
        self.fc_layer = nn.Sequential(
            nn.Linear((1**2 + 2**2 + 4**2) * 8 * base_dim, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.spp(x)
        x = self.fc_layer(x)
        return x