# model.py
import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
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

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        # CBR2d(Convolution, Batch Normalization, ReLU) 컴블루션 layer 설정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # 이미지나 데이터에서 특징을 뽑아냄
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
            # 각 레이어의 출력을 정규화(normalize)해서 학습을 더 빠르고 안정적으로 만드는 기법
            layers += [nn.BatchNorm2d(out_channels)]
            # 활성화 함수 ReLU
            layers += [nn.ReLU()]
            # 드롭아웃 추가
            layers += [nn.Dropout2d(p=0.3)]
            return nn.Sequential(*layers)

        # Contracting path -> 인코더 부분 : 이미지를 점점 압축하면서 특징 추출
        self.enc1_1 = CBR2d(1, 64)         # 인코더 블록 1: 1 → 64 채널
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)       # 크기 절반으로 줄이기

        self.enc2_1 = CBR2d(64, 128)       # 인코더 블록 2: 64 → 128 채널
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)      # 인코더 블록 3: 128 → 256 채널
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)      # 인코더 블록 4: 256 → 512 채널
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024)     # 인코더 블록 5: 512 →1024 채널

        # Expansive path -> 디코더 부분 : 해상도를 되살리면서 픽셀 수준의 예측을 수행
        self.dec5_1 = CBR2d(1024, 512)     # Bridge에서 나온 특징을 먼저 채널 축소 (1024 → 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, 2)  # 업샘플링: 해상도 2배 복원

        self.dec4_2 = CBR2d(1024, 512)     # enc4_2와 concat 후 conv (512*2 → 512)
        self.dec4_1 = CBR2d(512, 256)

        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, 2)  # 업샘플링: 해상도 2배 복원
        self.dec3_2 = CBR2d(512, 256)     # enc3_2와 concat 후 conv (256*2 → 256)
        self.dec3_1 = CBR2d(256, 128)

        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, 2)  # 업샘플링: 해상도 2배 복원
        self.dec2_2 = CBR2d(256, 128)     # enc2_2와 concat 후 conv (128*2 → 128)
        self.dec2_1 = CBR2d(128, 64)

        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, 2)    # 업샘플링: 해상도 2배 복원
        self.dec1_2 = CBR2d(128, 64)      # enc1_2와 concat 후 conv (64*2 → 64)
        self.dec1_1 = CBR2d(64, 64)

        self.fc = nn.Conv2d(64, 1, 1)     # 마지막 출력층: 64채널을 최종 출력 채널 1개로 줄임

    def forward(self, x):
        # 인코더
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

        # 디코더
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

        return self.fc(dec1_1)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice
