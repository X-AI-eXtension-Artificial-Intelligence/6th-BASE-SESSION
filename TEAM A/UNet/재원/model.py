import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g : 디코더 (gating signal) feature의 채널 수
        F_l : 인코더 (스킵 연결) feature의 채널 수
        F_int: 중간 채널 수 (보통 F_l의 절반 정도)
        """
        super(AttentionBlock, self).__init__()
        # 디코더 신호를 중간 차원으로 변환
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 인코더 feature를 중간 차원으로 변환
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 두 feature를 결합한 후 attention coefficient 계산
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        x: 인코더에서 나온 스킵 feature, 크기: [B, F_l, H, W]
        g: 디코더의 gating signal, 크기: [B, F_g, H, W] (보통 x와 동일한 spatial size)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, H, W]의 attention coefficient map
        return x * psi  # element-wise 곱하여 중요한 영역 강조

class AttentionUNet(nn.Module): 
    def __init__(self):
        super(AttentionUNet, self).__init__()
        
        # 기본 Convolution-BatchNorm-ReLU 블록
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                          padding=padding, dilation=dilation, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # 인코더 (Contracting path)
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64, dilation=1, padding=1)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64, dilation=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128, dilation=1, padding=1)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128, dilation=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256, dilation=2, padding=2)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256, dilation=2, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512, dilation=2, padding=2)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512, dilation=2, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024, dilation=2, padding=2)
        
        # 디코더 (Expansive path)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        # 각 레벨의 스킵 연결에 적용할 Attention Block
        # Level 4: 인코더의 enc4_2 (512 채널), gating: unpool4 (512 채널)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        # Level 3: 인코더의 enc3_2 (256 채널), gating: unpool3 (256 채널)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        # Level 2: 인코더의 enc2_2 (128 채널), gating: unpool2 (128 채널)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        # Level 1: 인코더의 enc1_2 (64 채널), gating: unpool1 (64 채널)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Decoder layers
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        # 인코더 단계
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
        
        # 디코더 단계
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        
        # Level 4: Attention 적용 후 스킵 연결
        att4 = self.att4(enc4_2, unpool4)
        cat4 = torch.cat((unpool4, att4), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        unpool3 = self.unpool3(dec4_1)

        # Level 3: Attention 적용
        att3 = self.att3(enc3_2, unpool3)
        cat3 = torch.cat((unpool3, att3), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        unpool2 = self.unpool2(dec3_1)

        # Level 2: Attention 적용
        att2 = self.att2(enc2_2, unpool2)
        cat2 = torch.cat((unpool2, att2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        unpool1 = self.unpool1(dec2_1)

        # Level 1: Attention 적용
        att1 = self.att1(enc1_2, unpool1)
        cat1 = torch.cat((unpool1, att1), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        out = self.fc(dec1_1)
        return out
