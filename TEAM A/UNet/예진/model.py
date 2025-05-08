# model.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ]
            return nn.Sequential(*layers)

        # ▼ 입력 채널을 1 → 3으로 변경 (흑백 → RGB)
        self.enc1_1 = CBR2d(3, 64)  # 변경됨: in_channels=1 → 3
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

        # ▼ 출력 채널을 1 → 10으로 변경 (이진 분류 → 다중 클래스 분류)
        self.fc = nn.Conv2d(64, 10, kernel_size=1)  # 변경됨: out_channels=1 → 10

    def forward(self, x):
        enc1 = self.enc1_2(self.enc1_1(x))
        enc2 = self.enc2_2(self.enc2_1(self.pool1(enc1)))
        enc3 = self.enc3_2(self.enc3_1(self.pool2(enc2)))
        enc4 = self.enc4_2(self.enc4_1(self.pool3(enc3)))
        enc5 = self.enc5_1(self.pool4(enc4))

        dec4 = self.unpool4(self.dec5_1(enc5))
        dec4 = self.dec4_1(self.dec4_2(torch.cat((dec4, enc4), dim=1)))

        dec3 = self.unpool3(dec4)
        dec3 = self.dec3_1(self.dec3_2(torch.cat((dec3, enc3), dim=1)))

        dec2 = self.unpool2(dec3)
        dec2 = self.dec2_1(self.dec2_2(torch.cat((dec2, enc2), dim=1)))

        dec1 = self.unpool1(dec2)
        dec1 = self.dec1_1(self.dec1_2(torch.cat((dec1, enc1), dim=1)))

        x = self.fc(dec1)
        return x
