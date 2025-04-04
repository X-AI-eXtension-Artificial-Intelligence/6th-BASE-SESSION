import torch
import torch.nn as nn

class ResidualCBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualCBR2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x = self.bn(x)
        x += residual
        return self.relu(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        self.num_classes = num_classes

        # Contracting path (Encoder with residual blocks)
        self.enc1_1 = ResidualCBR2d(1, 64)
        self.enc1_2 = ResidualCBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = ResidualCBR2d(64, 128)
        self.enc2_2 = ResidualCBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = ResidualCBR2d(128, 256)
        self.enc3_2 = ResidualCBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = ResidualCBR2d(256, 512)
        self.enc4_2 = ResidualCBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = ResidualCBR2d(512, 1024)

        # Expansive path (Decoder with residual blocks)
        self.dec5_1 = ResidualCBR2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.dec4_2 = ResidualCBR2d(1024, 512)
        self.dec4_1 = ResidualCBR2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.dec3_2 = ResidualCBR2d(512, 256)
        self.dec3_1 = ResidualCBR2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        self.dec2_2 = ResidualCBR2d(256, 128)
        self.dec2_1 = ResidualCBR2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.dec1_2 = ResidualCBR2d(128, 64)
        self.dec1_1 = ResidualCBR2d(64, 64)

        self.fc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
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

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        dec4_2 = self.dec4_2(torch.cat((unpool4, enc4_2), dim=1))
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        dec3_2 = self.dec3_2(torch.cat((unpool3, enc3_2), dim=1))
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        dec2_2 = self.dec2_2(torch.cat((unpool2, enc2_2), dim=1))
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        dec1_2 = self.dec1_2(torch.cat((unpool1, enc1_2), dim=1))
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x
