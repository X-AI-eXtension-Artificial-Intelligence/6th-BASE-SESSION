# 📁 Step 3: model.py ❤️
# U-Net 모델 구조 정의

import os
import numpy as np
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Conv + BN + ReLU (+ Dropout) 블록 생성 함수
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),  # 컨볼루션
                nn.BatchNorm2d(out_channels),                                                    # 배치 정규화
                nn.ReLU(inplace=True)                                                            # ReLU 활성화
            ]
            if dropout > 0.0: # 드롭아웃 인자 받으면 하게 함
                layers.append(nn.Dropout2d(dropout))                                             # 드롭아웃 (선택적으로)
            return nn.Sequential(*layers)

        # 인코더 블록: 이미지에서 특징 추출
        self.enc1_1 = CBR2d(1, 32)
        self.enc1_2 = CBR2d(32, 32)      # 64에서 32로 수정해봄
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(32, 64)      # 역시 1/2로 줄임
        self.enc2_2 = CBR2d(64, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(64, 128, dropout=0.1)  # 마지막 인코더 블록에 드롭아웃 추가
        self.pool3 = nn.MaxPool2d(2)

        # 디코더 블록: 특징 복원 및 업샘플링
        self.dec3_1 = CBR2d(128, 64) # 디코더 블록 추가
        self.unpool2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 업샘플링

        self.dec2_1 = CBR2d(128, 32)  # skip 연결 포함: 64(업샘플링) + 64(인코더)
        self.unpool1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.dec1_1 = CBR2d(64, 32)   # skip 연결 포함: 32(업샘플링) + 32(인코더)

        # 최종 출력 계층: 채널 수 1로 줄이고 Sigmoid로 이진값 생성
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # 인코더 경로
        enc1 = self.enc1_2(self.enc1_1(x))     # 인코더 블록 1
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))  # 인코더 블록 2
        enc3 = self.enc3_1(self.pool2(enc2))   # 인코더 블록 3

        # 디코더 경로
        dec3 = self.dec3_1(enc3)
        up2 = self.unpool2(dec3)              # 업샘플링 (2배)
        cat2 = torch.cat([up2, enc2], dim=1)  # 스킵 연결
        dec2 = self.dec2_1(cat2)

        up1 = self.unpool1(dec2)
        cat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1_1(cat1)

        # 최종 출력 (1채널 + Sigmoid로 이진 마스크 생성)
        out = self.final_conv(dec1)
        out = self.final_activation(out)

        return out
