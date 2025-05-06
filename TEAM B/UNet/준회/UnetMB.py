import torch
import torch.nn as nn
import torch.nn.functional as F

# 1) 채널 확장
# 2) Depthwise Conv 연산
# 3) SE Block
# 4) 채널 축

class MBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3, stride=1, se_ratio=0.3):
        super(MBConv2d, self).__init__()
        
        # 확장된 채널 개수
        mid_channels = in_channels * expand_ratio
        
        # 1x1 확장 Convolution
        self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise Convolution : 그룹 수를 mid_channels로 설정하여 채널별 독립적인 컨볼루션 수행
        # 이미지 줄어들지 말라고 padding = kernel_size // 2 적용
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size = kernel_size, stride = stride, padding = kernel_size//2, groups = mid_channels, bias = False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # SE (Squeeze-and-Excitation) Block
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 채널별 Gloval AVG Pooling 적용
            nn.Conv2d(mid_channels, se_channels, kernel_size=1), # SE Block에서 채널 축소
            nn.ReLU(), 
            nn.Conv2d(se_channels, mid_channels, kernel_size=1), # 채널 복원
            nn.Sigmoid() # 최종적으로 0 ~ 1 사이 가중치로 변환
        )
        
        # 1x1 Projection Convolution
        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # 1x1 확장 Conv -> BatchNorm -> ReLU 활성화 함수 적용
        Expansion = F.relu(self.bn1(self.expand_conv(x)))

        # Depthwise Convolution -> BatchNorm -> ReLU 활성화 함수 적용
        Depthwise = F.relu(self.bn2(self.depthwise_conv(Expansion)))
        
        # SE Block 적용 (채널별 가중치를 부여하는 과정)
        se_weight = self.se(Depthwise) # SE 블록에서 학습된 가중치 계산
        Depthwise_SE = Depthwise * se_weight # SE 가중치를 적용
        
        # 1x1 Projection Conv -> BatchNorm 적용
        Projection = self.bn3(self.project_conv(Depthwise_SE))
        
        return Projection

class UNetMB(nn.Module):
    def __init__(self):
        super(UNetMB, self).__init__()
        
        # Contracting path
        self.enc1_1 = MBConv2d(3, 64)
        self.enc1_2 = MBConv2d(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = MBConv2d(64, 128)
        self.enc2_2 = MBConv2d(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3_1 = MBConv2d(128, 256)
        self.enc3_2 = MBConv2d(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = MBConv2d(256, 512)
        self.enc4_2 = MBConv2d(512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = MBConv2d(512, 1024)

        ## 성능 향상을 위해서 한 번 더 MB Layer 추가
        self.enc5_2 = MBConv2d(1024, 1024)
        
        # Expansive path
        self.dec5_1 = MBConv2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        
        self.dec4_2 = MBConv2d(1024, 512)
        self.dec4_1 = MBConv2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        
        self.dec3_2 = MBConv2d(512, 256)
        self.dec3_1 = MBConv2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        
        self.dec2_2 = MBConv2d(256, 128)
        self.dec2_1 = MBConv2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.dec1_2 = MBConv2d(128, 64)
        self.dec1_1 = MBConv2d(64, 64)
        self.fc = nn.Conv2d(64, 2, kernel_size=1)
        
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
        enc5_2 = self.enc5_2(enc5_1)
        
        dec5_1 = self.dec5_1(enc5_2)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat([unpool4, enc4_2], dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat([unpool3, enc3_2], dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat([unpool2, enc2_2], dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat([unpool1, enc1_2], dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        fc = self.fc(dec1_1)
        return fc
