import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # CBR2d(Convolution, Batch Normalization, ReLU) 컴블루션 layer 설정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # 이미지나 데이터에서 특징을 뽑아냄
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # 각 레이어의 출력을 정규화(normalize)해서 학습을 더 빠르고 안정적으로 만드는 기법
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            # 활성화 함수 ReLU
            layers += [nn.ReLU()]

            # Sequential로 묶어서 하나의 블록처럼 만들기
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path -> 인코더 부분 : 이미지를 점점 압축하면서 특징 추출
        # channel = 1 -> 흑백

        # 인코더 블록  1: 1 → 64 채널
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 크기 절반으로 줄이기 (64x64 → 32x32 등)

        # 인코더 블록 2: 64 → 128 채널
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 크기 절반으로 줄이기

        # 인코더 블록 3: 128 → 256 채널
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2) # 크기 절반으로 줄이기

        # 인코더 블록 4: 256 → 512 채널
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2) # 크기 절반으로 줄이기

        # 인코더 블록 5: 512 →1024 채널
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        

        # Expansive path -> 디코더 부분 : 해상도를 되살리면서 픽셀 수준의 예측을 수행
        # 채널축소후 업샘플링링

        # Bridge에서 나온 특징을 먼저 채널 축소 (1024 → 512)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # 업샘플링: 512채널의 feature map을 2배 크기로 복원
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True) # 2x2 필터 , 2칸씩 이동 → 크기 2배
        
        # 이전 encoder의 enc4_2와 concat 후 conv (채널 512*2 → 512) -> 스킵연결결
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # 업샘플링: 256채널을 2배 크기로 복원
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # enc3_2와 concat 후 conv (256*2 → 256 → 128)
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # 업샘플링: 128채널 2배 확대
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # enc2_2와 concat 후 conv (128*2 → 128 → 64)
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # 업샘플링: 64채널 2배 확대
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # enc1_2와 concat 후 conv (64*2 → 64 → 64)
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # 마지막 출력층: 64채널을 최종 출력 채널 1개로 줄임
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        # 인코더
        enc1_1 = self.enc1_1(x)        # 1채널 → 64채널 conv
        enc1_2 = self.enc1_2(enc1_1)   # 64채널 유지 conv
        pool1 = self.pool1(enc1_2)     # 다운샘플링 (해상도 절반)

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

        dec5_1 = self.dec5_1(enc5_1)

        # 디코더
        unpool4 = self.unpool4(dec5_1)              # 512채널 upsample (해상도 2배)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # encoder의 enc4_2와 concat (skip connection)
        dec4_2 = self.dec4_2(cat4)                  # 채널 줄이는 conv
        dec4_1 = self.dec4_1(dec4_2)              

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

        return x
