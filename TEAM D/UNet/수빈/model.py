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
                                 bias=bias)]
            # Batch Normalization layer
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU layer
            layers += [nn.ReLU()]
            # layer들을 sequential하게 묶기
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # kernel_size, stride, padding, bias는 미리 설정해뒀으니 생략
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 입/출력 채널 개수 설정
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64) # 입/출력 채널 개수 설정

        self.pool1 = nn.MaxPool2d(kernel_size=2)             # Max pooling 진행(2X2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True) # up-convolution 진행(2X2)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) # Encoder 파트에서 전달된 512 채널이 입력으로 추가되기 때문
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

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
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1) # U-NET 구조 기준 가장 아래 있는 부분

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # Encoder 파트에서 전달된 채널과 합치는 부분
                                                   # 'dim=1'은 concat 방향을 의미(채널 방향으로 concat한다는 것이죠)
                                                   # 0: batch, 1:channel, 2: height(y방향), 3:width(x방향)라고 합니다

        dec4_2 = self.dec4_2(cat4)   # layer 전달 및 연결
        dec4_1 = self.dec4_1(dec4_2) # layer 전달 및 연결

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

        x = self.fc(dec1_1) # U-NET 구조 중 마지막 단계. 최종 출력 생성 단계. 


        return x  # logit 형태로 출력(logit: sigmoid/softmax 함수 적용 전. 후처리 과정에서 추가 연산 필요)
    