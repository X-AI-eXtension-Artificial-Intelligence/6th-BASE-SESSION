import os
import numpy as np

import torch
import torch.nn as nn

## U-Net 네트워크 정의
class UNet(nn.Module):
    def __init__(self):  # 필요한 레이어 선
        super(UNet, self).__init__()

        # Conv + BN + ReLU 묶어주는 함수 (계속 반복되니까 함수로)
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # 컨볼루션: 특징 추출
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # 배치 정규화: 학습 안정화
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU: 비선형성 추가
            layers += [nn.ReLU()]
            return nn.Sequential(*layers)

        ## 인코더 - 특징을 추출하고 점점 downsampling
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 1번째 stage의 1번째 화살표
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64) # 1번째 stage의 2번째 화살표
        self.pool1 = nn.MaxPool2d(kernel_size=2)             # 1/2로 줄어듦

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # bottleneck 
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)




        ## 디코더 (Expansive Path) - 업샘플링하면서 인코더 출력이랑 skip 연결

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)  # 채널 줄여주기

        # 업샘플링 (ConvTranspose로 크기 2배 늘림)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2)
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)  # in_channels : 인코더의 enc4_2랑 concat (채널 2배됨)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2)
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2)
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2)
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # 최종 출력 계층 1x1 Conv layer - 채널 수를 1로 줄여서 마스크(또는 예측 결과) 출력
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)








    def forward(self, x):   # 레이어 연결
        ## 인코더 경로 (downsampling) # Contracting path
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

        enc5_1 = self.enc5_1(pool4)  # bottleneck




        ## 디코더 경로 (upsampling) # Expansive path
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        # 인코더에서 나온 것과 concat (skip connection)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # dim = [0:batch, 1:channel, 2:height, 3:width]
        dec4_2 = self.dec4_2(cat4)
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

        # 최종 출력
        x = self.fc(dec1_1)

        return x
