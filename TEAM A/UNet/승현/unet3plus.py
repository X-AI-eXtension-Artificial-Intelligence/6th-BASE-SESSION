# unet3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ConvBlock(base_ch*8, base_ch*16)

        # Decoder aggregation
        def upconv(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.conv_cat = lambda ch: nn.Sequential(
            nn.Conv2d(ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(base_ch * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        # Decoder stage with multi-scale aggregation to x1 size
        x2_up = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x4_up = F.interpolate(x4, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x5_up = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x_cat = torch.cat([x1, x2_up, x3_up, x4_up, x5_up], dim=1)
        x_cat = self.conv_cat(x_cat.shape[1])(x_cat)
        out = self.out_conv(x_cat)

        return torch.sigmoid(out)
