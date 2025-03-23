import torch  
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# 1x1 컨볼루션
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 1x1 컨볼루션 연산 수행 (입력 채널 수, 출력 채널 수)
        nn.ReLU() ,
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model 

# 3x3 컨볼루션 
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 
        nn.ReLU(),  # 활성화 함수 ReLU
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  
        nn.ReLU(),  
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  
        nn.ReLU(),  
        nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링으로 맵 크기 감소
    )
    return model  

# VGG16 모델 클래스 
class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=10):  # 클래스 수 입력 받기, 모델 생성
        super(VGG16, self).__init__()  
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), #64
            conv_2_block(base_dim,2*base_dim), #128
            conv_3_block(2*base_dim,4*base_dim), #256
            conv_3_block(4*base_dim,8*base_dim), #512
            conv_3_block(8*base_dim,8*base_dim), #512        
        )
        self.fc_layer = nn.Sequential( 
            nn.Linear(8*base_dim*1*1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    