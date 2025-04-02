import os
import numpy as np
import torch
import torch.nn as nn

## UNet 모델 클래스 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Conv → BatchNorm → ReLU 블록을 정의하는 함수
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            return nn.Sequential(*layers)

        # Contracting path (인코더)
        self.enc1_1 = CBR2d(1, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024)  # bottleneck 부분

        # Expansive path (디코더)
        self.dec5_1 = CBR2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(1024, 512)  # 512(dec5) + 512(enc4)
        self.dec4_1 = CBR2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(512, 256)  # 256(dec4) + 256(enc3)
        self.dec3_1 = CBR2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(256, 128)  # 128(dec3) + 128(enc2)
        self.dec2_1 = CBR2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.dec1_2 = CBR2d(128, 64)  # 64(dec2) + 64(enc1)
        self.dec1_1 = CBR2d(64, 64)

        self.fc = nn.Conv2d(64, 1, kernel_size=1)  # 마지막 출력층 (1채널 마스크)

    ## forward 연산 정의
    def forward(self, x):
        # 인코더
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

        # 디코더
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # skip connection
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

        x = self.fc(dec1_1)  # 최종 출력 (1채널)

        return x
