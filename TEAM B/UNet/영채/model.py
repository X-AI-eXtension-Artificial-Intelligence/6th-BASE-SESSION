import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Conv -> BatchNorm -> ReLU
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        # 입력에서 중요한 특징을 추출하며 점점 작게
        # 단계마다 2개의 convolution 블록을 거친 후 max pooling 수행

        # 채널 수 1 -> 64 
        self.enc1_1 = CBR2d(1, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)  

        # 64 -> 128
        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        # 128 -> 256
        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        # 256 -> 512
        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024)

        #디코더
        #ConvTranspose2d는 업샘플링을 수행

        self.dec5_1 = CBR2d(1024, 512)           
        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, 2)

        self.dec4_2 = CBR2d(1024, 512)               
        self.dec4_1 = CBR2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, 2)

        self.dec3_2 = CBR2d(512, 256)              
        self.dec3_1 = CBR2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.dec2_2 = CBR2d(256, 128)               
        self.dec2_1 = CBR2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.dec1_2 = CBR2d(128, 64)               
        self.dec1_1 = CBR2d(64, 64)

        # 채널 수 64 -> 1
        self.fc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        #인코더 단계
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

        #Bottleneck 
        enc5_1 = self.enc5_1(pool4)

        #디코더 단계
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x
