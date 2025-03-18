# vgg-16 architecture
# 모든 convNet은 kernel_size=3x3, stride=1, padding=1로 고정
'''
- 3x3 CONV layer x2 (filter_size 64)
- 3x3 CONV layer x2 (filter_size 128)
- 3x3 CONV layer x3 (filter_size 256)
- 3x3 CONV layer x3 (filter_size 512)
- 3x3 CONV layer x3 (filter_size 512)
- FC layer 4096
- FC layer 4096
- FC layer 1000
'''

import torch.nn as nn
import pandas as pd
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

class VGG16(nn.Module):
    '''
    model class를 정의할 때 nn.Module을 상속받는 이유?
    nn.Module : 딥러닝 모델을 구현할 때 필요한 기능(계층 관리, 파라미터 추적, 학습/평가 모드 전환, 저장/로드 등)을 통합적으로 제공
    '''
    def __init__(self, base_dim, num_classes=10): # CIFAR10 사용했으므로
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 64
            conv_2_block(base_dim,2*base_dim), # 128
            conv_3_block(2*base_dim, 4*base_dim), # 256
            conv_3_block(4*base_dim, 8*base_dim), # 512
            conv_3_block(8*base_dim, 8*base_dim), # 512
        )
        '''
        CIFAR10 (32×32) 이미지를 사용하면 여러 차례의 conv와 pooling을 거쳐 최종 피처맵 크기가 1×1이 됨
        => fc_layer에 입력되는 피쳐의 총 개수: (채널 수)×(가로 크기)×(세로 크기)로 계산되므로 8×base_dim×1×1이 되는 것
        논문처럼 입력 이미지가 224×224이면, V마지막 conv layer 후 피처맵의 크기가 보통 7×7이 되어 7×7×채널 수가 입력 차원이 됨
        '''
        self.fc_layer = nn.Sequential(
            # CIFAR10은 크기가 32x32이므로
            nn.Linear(8*base_dim*1*1, 4096), # 왜 1*1 사용 ?
            nn.ReLU(True), # in-place 연산 True 의미. 텐서를 새롭게 생성하지 않고 기존 메모리를 덮어쓰는 방식으로 연산을 수행
            nn.Dropout(),
            nn.Linear(4096, 10),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        '''
        cf). .view(배치사이즈, -1) 사용 이유
        파라미터 -1은 PyTorch가 나머지 차원을 자동으로 계산하도록 함. 즉, 나머지 모든 요소를 한 차원으로 합침
        FC layer는 2차원을 받기 때문에 conv layer를 지나서 fc layer에 들어가기 전에 차원을 축소한 것
        '''
        # print(x.shape)
        x = self.fc_layer(x)
        return x