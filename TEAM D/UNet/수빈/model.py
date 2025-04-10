# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn

# U-NET 네트워크 구축  
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Convolution/Batch Normalization/ReLU 함수를 조합해서 하나의 레이어로 만드는 과정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # 레이어 저장할 리스트 생성
            layers = []
            # 2D Convolution layer
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]\
                                 
            ### BatchNorm → GroupNorm ###
            # 기존 BatchNorm은 배치 단위로 평균/분산 계산하여 정규화.
            # batch_size가 크면 더 정확하게 정규화되었음.
            # GroupNorm은 batch_size에 의존하지 않고, channel 단위로 그룹을 나누어서 정규화 함
            # 하단에서는 num_groups=4로 정했고, out_channels를 4개 group으로 나누어서 그 그룹에서 평균/분산 계산하여 정규화하는 방식
            # 메모리 사용량이 낮아서 모델 경량화에 좋음
            # 속도는 좀 더 느리지만, 학습이 전체적으로 안정적이라고 함
            layers += [nn.GroupNorm(num_groups=4, num_channels=out_channels)] 
            # ReLU layer
            layers += [nn.ReLU()]
            # layer들을 sequential하게 묶기
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # kernel_size, stride, padding, bias는 미리 설정해뒀으니 생략
        self.enc1_1 = CBR2d(in_channels=1, out_channels=32)  # 입/출력 채널 개수 설정
        self.enc1_2 = CBR2d(in_channels=32, out_channels=32) # 입/출력 채널 개수 설정

        self.pool1 = nn.MaxPool2d(kernel_size=2)             # Max pooling 진행(2X2)

        self.enc2_1 = CBR2d(in_channels=32, out_channels=64)
        self.enc2_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc3_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=128, out_channels=256)

        # Expansive path
        self.dec4_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)

        self.dec3_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec3_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.dec2_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec2_1 = CBR2d(in_channels=64, out_channels=32)
        self.unpool1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)

        self.dec1_2 = CBR2d(in_channels=2 * 32, out_channels=32)
        self.dec1_1 = CBR2d(in_channels=32, out_channels=32)

        self.fc = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # 각 레이어들 연결
    def forward(self, x):
        enc1_1 = self.enc1_1(x)      # layer 전달 및 연결
        enc1_2 = self.enc1_2(enc1_1) # layer 전달 및 연결
        pool1 = self.pool1(enc1_2)   # Pooling을 통해 size 줄이기

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)

        dec4_1 = self.dec4_1(enc4_1)
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)

        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)

        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)

        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x  # logit 형태로 출력(logit: sigmoid/softmax 함수 적용 전. 후처리 과정에서 추가 연산 필요)