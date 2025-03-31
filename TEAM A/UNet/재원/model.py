# 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn


# 네트워크 구조
class UNet(nn.Module): 
    def __init__(self):
        super(UNet, self).__init__() #생성자 초기화

        # Convolution + 배치정규화 + 활성화함수 블록 정의(U-Net에서 한층에 두번씩 쓰이는 기본 Conv연산)
        # input channel, output channel, 커널 사이즈 3*3, stride 1로 기본값 지정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True): 
            cbr = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), #2d Conv연산
                    nn.BatchNorm2d(num_features=out_channels), # 배치 정규화는 나온 output에 적용
                    nn.ReLU()
                    )

            return cbr

        #수축 경로(Contracting path)
        
        #Conv연산 2 layer
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64) #초기 grayscale 이미지 -> input channel 1로 지정
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        #2*2 Max pooling으로 다운 샘플링
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        #Conv연산 2 layer
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        #2*2 Max pooling으로 다운 샘플링
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        #Conv연산 2 layer
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        #2*2 Max pooling으로 다운 샘플링
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        #Conv연산 2 layer
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        #2*2 Max pooling으로 다운 샘플링
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        #최하단에 해당하는 Bottleneck 설계(다운샘플링 없이 Conv 연산만 1번)
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        #확장 경로(Expansive path)
        
        #업샘플링 전 한번 더 Bottleneck Conv 연산
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        #transpose conv
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        #Skip connection -> 차원 수 2배 되기 때문에 입력 채널 수 2배
        #일반 3*3 Conv 2번 
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        #transpose conv
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        #일반 3*3 Conv 2번 
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        #transpose conv
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        #일반 3*3 Conv 2번 
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        #transpose conv
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        #일반 3*3 Conv 2번 
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)


        #기존 논문에서는 최종 layer output 채널 수 2인데, 데이터셋이 배경/객체 이진 분류라 1로 조정
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
    
    # 순전파 함수 설정(위에서 만들어 놓은 것 연결)
    def forward(self, x):
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

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
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

        x = self.fc(dec1_1)

        return x