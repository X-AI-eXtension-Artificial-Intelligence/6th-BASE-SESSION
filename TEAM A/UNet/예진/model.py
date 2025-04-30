# UNet: 이미지를 입력하면 이미지 안에 무엇이 어디에 있는지 구분해주는 딥러닝 모델
# 이미지 한 장을 넣고, 픽셀 단위로 뭐가 뭔지 구분해서 다시 이미지로 뱉어내야 함!
# 입력 이미지를 -> 작게 줄여가면서 -> 특징을 뽑고 -> 다시 키우면서 원래 크기로 복원하는 구조

import os
import numpy as np

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 인코더
        self.enc1 = nn.Sequential(CBR2d(1, 64), CBR2d(64, 64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(CBR2d(64, 128), CBR2d(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(CBR2d(128, 256), CBR2d(256, 256))
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(CBR2d(256, 512), CBR2d(512, 512))
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = CBR2d(512, 1024)

        # 디코더
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(CBR2d(1024, 512), CBR2d(512, 256))

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(CBR2d(512, 256), CBR2d(256, 128))

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR2d(256, 128), CBR2d(128, 64))

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR2d(128, 64), CBR2d(64, 64))

        self.final = nn.Conv2d(64, 3, kernel_size=1)  # 클래스 수 3

    def forward(self, x):
        enc1 = self.enc1(x)       # [B, 64, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))  # [B, 512, H/8, W/8]
        enc5 = self.enc5(self.pool4(enc4))  # [B, 1024, H/16, W/16]

        dec4 = self.up4(enc5)                # [B, 512, H/8, W/8]
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.up3(dec4)                # [B, 256, H/4, W/4]
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)                # [B, 128, H/2, W/2]
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)                # [B, 64, H, W]
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return self.final(dec1)              # [B, 3, H, W]
