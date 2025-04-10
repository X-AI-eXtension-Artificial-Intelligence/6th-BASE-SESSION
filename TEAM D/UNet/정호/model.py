import torch
import torch.nn as nn
import torch.nn.functional as F

# CBR(Conv2d + BatchNorm2d + ReLU) 블록 정의
# 기존의 U-Net에서 Batchnorm 추가, He초기화 적용
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(CBR2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)  # LeakyReLU → ReLU로 변경 (원래 U-Net처럼)

        # He 초기화 적용
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UNet_Improved(nn.Module):
    def __init__(self):
        super(UNet_Improved, self).__init__()

        # 인코더 (Residual 연결을 위한 구조로 변경)
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

        self.enc5_1 = CBR2d(512, 1024)

        # 디코더
        self.dec5_1 = CBR2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.dec4_2 = CBR2d(1024, 512)
        self.dec4_1 = CBR2d(512, 256)

        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 128)

        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 64)

        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)

        self.fc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 인코더
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        enc1_out = enc1_2 + enc1_1  # Residual 연결 추가
        pool1 = self.pool1(enc1_out)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        enc2_out = enc2_2 + enc2_1  # Residual 연결 추가
        pool2 = self.pool2(enc2_out)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        enc3_out = enc3_2 + enc3_1  # Residual 연결 추가
        pool3 = self.pool3(enc3_out)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        enc4_out = enc4_2 + enc4_1  # Residual 연결 추가
        pool4 = self.pool4(enc4_out)

        enc5_1 = self.enc5_1(pool4)

        # 디코더
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_out), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_out), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_out), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_out), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        out = self.fc(dec1_1)

        return out
