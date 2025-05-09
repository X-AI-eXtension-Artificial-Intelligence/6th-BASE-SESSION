import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1_1 = CBR2d(3, 64)     # 입력 채널 3
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

        # Bottleneck
        self.bottleneck1 = CBR2d(512, 1024)
        self.bottleneck2 = CBR2d(1024, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4_1 = CBR2d(1024, 512)
        self.dec4_2 = CBR2d(512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_1 = CBR2d(512, 256)
        self.dec3_2 = CBR2d(256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_1 = CBR2d(256, 128)
        self.dec2_2 = CBR2d(128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_1 = CBR2d(128, 64)
        self.dec1_2 = CBR2d(64, 64)

        self.fc = nn.Conv2d(64, 32, 1)  # CamVid 클래스 수만큼 출력

    def forward(self, x):
        # Encoder
        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck2(self.bottleneck1(pool4))

        # Decoder (skip connection 전에 크기 맞추기)
        up4 = self.upconv4(bottleneck)
        if up4.shape != enc4.shape:
            diffY = enc4.size()[2] - up4.size()[2]
            diffX = enc4.size()[3] - up4.size()[3]
            up4 = F.pad(up4, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4_2(self.dec4_1(merge4))

        up3 = self.upconv3(dec4)
        if up3.shape != enc3.shape:
            diffY = enc3.size()[2] - up3.size()[2]
            diffX = enc3.size()[3] - up3.size()[3]
            up3 = F.pad(up3, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3_2(self.dec3_1(merge3))

        up2 = self.upconv2(dec3)
        if up2.shape != enc2.shape:
            diffY = enc2.size()[2] - up2.size()[2]
            diffX = enc2.size()[3] - up2.size()[3]
            up2 = F.pad(up2, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2_2(self.dec2_1(merge2))

        up1 = self.upconv1(dec2)
        if up1.shape != enc1.shape:
            diffY = enc1.size()[2] - up1.size()[2]
            diffX = enc1.size()[3] - up1.size()[3]
            up1 = F.pad(up1, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1_2(self.dec1_1(merge1))

        out = self.fc(dec1)
        return out
