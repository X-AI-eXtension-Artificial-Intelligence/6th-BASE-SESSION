#모듈 불러오기
import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):  # nn.Module을 상속하여 UNet 클래스 정의
    def __init__(self):
        super(UNet, self).__init__()  # 부모 클래스의 초기화 함수 호출

        # 기본 컨볼루션 블록을 정의하는 함수
        #2D 컨볼루션 + 배치 정규화 + ReLU 활성화 함수를 포함하는 블럭
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):

            layers = []  # 레이어 리스트 초기화
            
            # 2D 컨볼루션 레이어 추가
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # 배치 정규화 추가
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU 활성화 함수 추가
            layers += [nn.ReLU()]

            # nn.Sequential을 사용하여 하나의 블록으로 묶어서 반환
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(1, 64)  # 입력(1채널) → 64채널 컨볼루션
        self.enc1_2 = CBR2d(64, 64)  # 64채널 → 64채널 컨볼루션
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 크기를 절반으로 줄이는 풀링 연산

        self.enc2_1 = CBR2d(64, 128)  # 64 → 128
        self.enc2_2 = CBR2d(128, 128)  # 128 → 128
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 풀링 (절반 축소)

        self.enc3_1 = CBR2d(128, 256)  # 128 → 256
        self.enc3_2 = CBR2d(256, 256)  # 256 → 256
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 풀링

        self.enc4_1 = CBR2d(256, 512)  # 256 → 512
        self.enc4_2 = CBR2d(512, 512)  # 512 → 512
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # 풀링

        self.enc5_1 = CBR2d(512, 1024)  # 가장 깊은 층 (보틀넥) 512 → 1024

        # Expansive path
        self.dec5_1 = CBR2d(1024, 512)  # 1024 → 512
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0)  # 업샘플링 (2배 확대)

        self.dec4_2 = CBR2d(1024, 512)  # 512 + 512(Skip Connection) → 512
        self.dec4_1 = CBR2d(512, 256)  # 512 → 256
        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0)  # 업샘플링

        self.dec3_2 = CBR2d(512, 256)  # 256 + 256 → 256
        self.dec3_1 = CBR2d(256, 128)  # 256 → 128
        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)  # 업샘플링

        self.dec2_2 = CBR2d(256, 128)  # 128 + 128 → 128
        self.dec2_1 = CBR2d(128, 64)  # 128 → 64
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)  # 업샘플링

        self.dec1_2 = CBR2d(128, 64)  # 64 + 64 → 64
        self.dec1_1 = CBR2d(64, 64)  # 64 → 64
        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)  # 최종 출력층 (1x1 컨볼루션, 출력 채널: 1)

    def forward(self, x):
        """ 순전파(Forward Propagation) 과정 정의 """
        #위에서 나온 self.~~를 적용하는 부분
        # 🔹 **인코딩 경로 (Contracting Path)**
        enc1_1 = self.enc1_1(x)  
        enc1_2 = self.enc1_2(enc1_1)  
        pool1 = self.pool1(enc1_2)  

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

        # 🔹 **디코딩 경로 (Expansive Path)**
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)

        dec4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4 = self.dec4_2(dec4)
        dec4 = self.dec4_1(dec4)
        unpool3 = self.unpool3(dec4)

        dec3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3 = self.dec3_2(dec3)
        dec3 = self.dec3_1(dec3)
        unpool2 = self.unpool2(dec3)

        dec2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2 = self.dec2_2(dec2)
        dec2 = self.dec2_1(dec2)
        unpool1 = self.unpool1(dec2)

        dec1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1 = self.dec1_2(dec1)
        dec1 = self.dec1_1(dec1)

        x = self.fc(dec1)  # 최종 출력층 적용
        return x