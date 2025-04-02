## 모델만 저장하는 파일

import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            ## kernel_size=3, stride=1, padding=1, bias=True -> 변하지 X
            ## 논문에서 하는 것처럼 패딩을 아예 없애면 나중에 크롭해야 하기 때문에 크롭하는 과정을 없애기 위해 padding=1로 설정
            layers = []
            ## 컨볼루션 레이어 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            ## 배치 노말라이제이션 레이어 정의
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            ## ReLU 레이어 정의
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path -> out_channels가 2배씩 늘어남
        ## 첫번째 인코더
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        ## max pooling 레이어
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        ## 두번째 인코더
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        ## 세번째 인코더
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        ## 네번째 인코더
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        ## 다섯번째 인코더
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)


        # Expansive path -> out_channels가 1/2이 됨
        ## ConvTranspose2d을 통해 업 컨볼루션 진행
        ## 다섯번째 디코더 -> 5번째 인코더와 정확히 반대됨
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        ## 네번째 디코더
        ## 업 컨볼루션 2*2 
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        ## enc4_2의 out_channels과 dec4_2의 in_channels이 512로 같으면 안 됨
        ## Because 구조를 보면 dec4_2의 아래 채널에서 올라온 512와 enc4_2에서 온 512가 합쳐져서 512 * 2가 되어야 함
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        ## 세번째 디코더
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        ## dec3_2의 in_channels은 그냥 256이 아니라 스킵 커넥션으로 오는 256이 있기 때문에 2배
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        ## 두번째 디코더
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        ## 첫번째 디코더
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        ## 마지막으로 세그멘테이션에 필요한 n개의 클래스에 대한 아웃풋을 만들어주기 위해 1x1 컨볼루션을 추가
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    ## 순전파
    def forward(self, x):
        ## 인코더
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


        ## 디코더
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        ## 스킵커넥션과 그대로 전달 받은 것을 cat
        ## 참고: dim = [0: batch 방향, 1: channel 방향, 2: height (y) 방향, 3: width (x) 방향]
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
