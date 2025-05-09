import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(3, 32), CBR(32, 32))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(32, 64), CBR(64, 64))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool3 = nn.MaxPool2d(2)
        self.center = CBR(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(128, 64), CBR(64, 32))
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(64, 32), CBR(32, 32))
        self.final = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        c = self.center(self.pool3(e3))
        d3 = self.up3(c)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return self.sigmoid(out)
