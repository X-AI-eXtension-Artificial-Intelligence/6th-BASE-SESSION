import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_g, in_x, inter_channels):
        super().__init__()
        # 1x1 conv 
        self.W_g = nn.Sequential(
            nn.Conv2d(in_g, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_x, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels*2, base_channels*4, dilation=2)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_channels*4, base_channels*8, dilation=2)
        self.pool4 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(base_channels*8, base_channels*16, dilation=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(in_g=base_channels*8, in_x=base_channels*8, inter_channels=base_channels*4)
        self.dec4 = DoubleConv(base_channels*16, base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(in_g=base_channels*4, in_x=base_channels*4, inter_channels=base_channels*2)
        self.dec3 = DoubleConv(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(in_g=base_channels*2, in_x=base_channels*2, inter_channels=base_channels)
        self.dec2 = DoubleConv(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(in_g=base_channels, in_x=base_channels, inter_channels=base_channels//2)
        self.dec1 = DoubleConv(base_channels*2, base_channels)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        x4 = self.down4(p3)
        p4 = self.pool4(x4)
        xb = self.bottom(p4)

        # Decoder path with attention skip
        u4 = self.up4(xb)
        a4 = self.att4(x4, u4)
        c4 = torch.cat([u4, a4], dim=1)
        d4 = self.dec4(c4)

        u3 = self.up3(d4)
        a3 = self.att3(x3, u3)
        c3 = torch.cat([u3, a3], dim=1)
        d3 = self.dec3(c3)

        u2 = self.up2(d3)
        a2 = self.att2(x2, u2)
        c2 = torch.cat([u2, a2], dim=1)
        d2 = self.dec2(c2)

        u1 = self.up1(d2)
        a1 = self.att1(x1, u1)
        c1 = torch.cat([u1, a1], dim=1)
        d1 = self.dec1(c1)

        out = self.out_conv(d1)
        return out
