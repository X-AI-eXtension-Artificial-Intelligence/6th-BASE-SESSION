import torch
import torch.nn as nn

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Residual Block
class ResidualCBR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualCBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        res = self.residual(x)
        return self.relu(out + res)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1_1 = ResidualCBR(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ResidualCBR(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ResidualCBR(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = ResidualCBR(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = ResidualCBR(256, 512)

        # Expansive path
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = ResidualCBR(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = ResidualCBR(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = ResidualCBR(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(32, 32, 16)
        self.dec1 = ResidualCBR(64, 32)

        self.fc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.enc1_1(x)
        e2 = self.enc2_1(self.pool1(e1))
        e3 = self.enc3_1(self.pool2(e2))
        e4 = self.enc4_1(self.pool3(e3))
        e5 = self.enc5_1(self.pool4(e4))

        # Decoding path
        d4 = self.up4(e5)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.fc(d1)
        return out

# import os
# import numpy as np

# import torch
# import torch.nn as nn

# ## 네트워크 구축하기
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#             layers = []
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                  kernel_size=kernel_size, stride=stride, padding=padding,
#                                  bias=bias)]
#             layers += [nn.BatchNorm2d(num_features=out_channels)]
#             layers += [nn.ReLU()]

#             cbr = nn.Sequential(*layers)

#             return cbr

#         # Contracting path
#         self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
#         self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

#         self.pool1 = nn.MaxPool2d(kernel_size=2)

#         self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
#         self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
#         self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

#         self.pool3 = nn.MaxPool2d(kernel_size=2)

#         self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
#         self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

#         self.pool4 = nn.MaxPool2d(kernel_size=2)

#         self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

#         # Expansive path
#         self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

#         self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
#         self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

#         self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
#         self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

#         self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
#         self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

#         self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
#         self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

#         self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
#         enc1_1 = self.enc1_1(x)
#         enc1_2 = self.enc1_2(enc1_1)
#         pool1 = self.pool1(enc1_2)

#         enc2_1 = self.enc2_1(pool1)
#         enc2_2 = self.enc2_2(enc2_1)
#         pool2 = self.pool2(enc2_2)

#         enc3_1 = self.enc3_1(pool2)
#         enc3_2 = self.enc3_2(enc3_1)
#         pool3 = self.pool3(enc3_2)

#         enc4_1 = self.enc4_1(pool3)
#         enc4_2 = self.enc4_2(enc4_1)
#         pool4 = self.pool4(enc4_2)

#         enc5_1 = self.enc5_1(pool4)

#         dec5_1 = self.dec5_1(enc5_1)

#         unpool4 = self.unpool4(dec5_1)
#         cat4 = torch.cat((unpool4, enc4_2), dim=1)
#         dec4_2 = self.dec4_2(cat4)
#         dec4_1 = self.dec4_1(dec4_2)

#         unpool3 = self.unpool3(dec4_1)
#         cat3 = torch.cat((unpool3, enc3_2), dim=1)
#         dec3_2 = self.dec3_2(cat3)
#         dec3_1 = self.dec3_1(dec3_2)

#         unpool2 = self.unpool2(dec3_1)
#         cat2 = torch.cat((unpool2, enc2_2), dim=1)
#         dec2_2 = self.dec2_2(cat2)
#         dec2_1 = self.dec2_1(dec2_2)

#         unpool1 = self.unpool1(dec2_1)
#         cat1 = torch.cat((unpool1, enc1_2), dim=1)
#         dec1_2 = self.dec1_2(cat1)
#         dec1_1 = self.dec1_1(dec1_2)

#         x = self.fc(dec1_1)

#         return x
