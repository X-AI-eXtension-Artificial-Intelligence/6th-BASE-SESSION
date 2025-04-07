import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # CBR블록 정의 . conv, BN, RelU로 구성된 작은 블록 
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers) # 위 3개의 레이어를 묶어서 하나의 블록처럼 사용할 수 있게 해

            return cbr

        # Contracting path. 수축경로. layer 두 개마다 pooling을 거쳐 1024채널(1-64-128-256-512-1024)까지 사이즈 수축 
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64) 
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

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
        self.enc5_2 = CBR2d(in_channels=1024, out_channels=1024)

        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        self.enc6_1 = CBR2d(in_channels=1024, out_channels=2048)  
        


        # Expansive path. 확장 경로 
        self.dec6_1 = CBR2d(in_channels=2048, out_channels=1024)
        self.unpool5 = nn.ConvTranspose2d(in_channels=1024,
                                        out_channels=1024, 
                                        kernel_size=2, 
                                        stride=2,
                                        padding=0,
                                        bias=True)

        self.dec5_2 = CBR2d(in_channels=2 * 1024, out_channels=1024)  # skip 연결 포함
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)  # 인코더 최종 출력 1024채널을 512채널로 줄임 

        self.unpool4 = nn.ConvTranspose2d(  # 공간적인 크기를 늘리는(convolution의 반대) 연산을 해주는 합성곱 레이어
                                          in_channels=512,  # 입력 채널 
                                          out_channels=512,  # 출력 채널 
                                          kernel_size=2,  # 커널 크기 
                                          stride=2,  # 업샘플링 비율. 2배 확대 
                                          padding=0, 
                                          bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)  # 위에서 가져온 512채널과 인코더에서 가져온 512채널 합 
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)  # 256채널 까지 줄임 

# 업샘플링, 스킵 커넥으로 2배 늘리고 절반, 절반
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

# 업샘플링, 스킵 커넥으로 2배 늘리고 절반, 절반
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

# 업샘플링, 스킵 커넥으로 2배 늘리고 절반, 절반
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

# 1x1 conv로 채널 수를 원하는 클래스 수로 맞추기. 배경-객체 이진 분류 이므로 채널 1
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)



    def forward(self, x):  # 순전파 
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
        enc5_2 = self.enc5_2(enc5_1)
        pool5 = self.pool5(enc5_2)

        enc6_1 = self.enc6_1(pool5)  # 수축 끝 


        dec6_1 = self.dec6_1(enc6_1)  # 확장 시작 

        unpool5 = self.unpool5(dec6_1)
        cat5 = torch.cat((unpool5, enc5_2), dim=1)
        dec5_2 = self.dec5_2(cat5)
        dec5_1 = self.dec5_1(enc5_1)  

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # pooling결과와, 인코더에서 skip connec 가져오기 
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
