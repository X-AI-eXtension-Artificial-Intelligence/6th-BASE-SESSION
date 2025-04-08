import torch
import torch.nn as nn

# Conv + BatchNorm + ReLU (+ Dropout) 블록
def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dropout_p=0.0):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU()
    ]
    if dropout_p > 0:
        layers.append(nn.Dropout2d(p=dropout_p))  # 드롭아웃 추가 (p>0일 때만)
    
    return nn.Sequential(*layers)

# UNet 모델
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 인코더 (Contracting path)
        self.enc1_1 = CBR2d(1, 64, dropout_p=0.1)
        self.enc1_2 = CBR2d(64, 64, dropout_p=0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128, dropout_p=0.1)
        self.enc2_2 = CBR2d(128, 128, dropout_p=0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256, dropout_p=0.2)
        self.enc3_2 = CBR2d(256, 256, dropout_p=0.2)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512, dropout_p=0.3)
        self.enc4_2 = CBR2d(512, 512, dropout_p=0.3)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024, dropout_p=0.3)

        # 디코더 (Expansive path)
        self.unpool4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4_2 = CBR2d(1024, 512, dropout_p=0.3)
        self.dec4_1 = CBR2d(512, 256, dropout_p=0.2)

        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3_2 = CBR2d(512, 256, dropout_p=0.2)
        self.dec3_1 = CBR2d(256, 128, dropout_p=0.1)

        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2_2 = CBR2d(256, 128, dropout_p=0.1)
        self.dec2_1 = CBR2d(128, 64, dropout_p=0.1)

        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1_2 = CBR2d(128, 64, dropout_p=0.1)
        self.dec1_1 = CBR2d(64, 64, dropout_p=0.1)

        self.fc = nn.Conv2d(64, 1, kernel_size=1)  # 최종 1채널로 output

    def forward(self, x):
        # 인코더
        enc1 = self.enc1_1(x)
        enc1 = self.enc1_2(enc1)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_1(pool1)
        enc2 = self.enc2_2(enc2)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_1(pool2)
        enc3 = self.enc3_2(enc3)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_1(pool3)
        enc4 = self.enc4_2(enc4)
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        # 디코더
        dec4 = self.unpool4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4_2(dec4)
        dec4 = self.dec4_1(dec4)

        dec3 = self.unpool3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3_2(dec3)
        dec3 = self.dec3_1(dec3)

        dec2 = self.unpool2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2_2(dec2)
        dec2 = self.dec2_1(dec2)

        dec1 = self.unpool1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1_2(dec1)
        dec1 = self.dec1_1(dec1)

        out = self.fc(dec1)
        return out
