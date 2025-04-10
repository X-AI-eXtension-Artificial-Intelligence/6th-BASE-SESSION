# 1. 업샘플링 방식 변경: ConvTranspose2d → Upsample + Conv2d
# 
# - 기존 U-Net은 디코더 경로에서 feature map의 해상도를 복원할 때 ConvTranspose2d를 사용함.
# - ConvTranspose2d는 학습 가능한 파라미터를 가지며, 학습 중에 spatial artifact (checkerboard effect)가 발생할 수 있음.
# - 이는 특히 경계선이 중요한 의료 영상 segmentation에서 출력의 신뢰도를 떨어뜨릴 수 있음.
# 
# → 개선: nn.Upsample(mode='bilinear')을 사용하여 해상도를 먼저 선형 보간으로 복원한 후, Conv2d를 통해 채널 정보를 재정렬함.
# → 결과적으로 복원된 mask가 더 부드럽고 안정적이며, artifact 감소 효과가 있음.
# 
# 이와 같은 구조는 최근에 발표된 lightweight segmentation 네트워크들에서도 주로 채택되고 있음.

# 2. 과적합 방지를 위한 Dropout 도입
#
# - U-Net은 encoder path에서 feature depth가 깊어지며 파라미터 수가 급격히 증가함.
# - 특히 의료용 segmentation task는 데이터셋이 작기 때문에 학습 중 과적합이 쉽게 발생함.
#
# → encoder 후반부 (enc4)와 bottleneck에 Dropout(p=0.5)을 추가하여 과적합을 방지함.
# → 학습 중 일부 뉴런을 비활성화함으로써 feature에 대한 모델의 과도한 의존을 줄이고, 일반화 성능을 향상시킴.
#
# 이 구성은 U-Net 변형 모델(예: Attention U-Net, UNet++ 등)에서도 공통적으로 사용되는 regularization 기법임.

# 3. 출력 채널 수 유연화 (multi-class 대응)
#
# - 기존 U-Net은 binary segmentation만을 염두에 두고 마지막 layer의 out_channels=1로 고정되어 있음.
#
# → output layer를 Conv2d(64, n_classes, kernel_size=1)로 설정하여 유연하게 출력 클래스 수를 설정할 수 있도록 개선함.
# → binary일 경우 n_classes=1, multi-class segmentation의 경우 n_classes>1로 쉽게 확장 가능.
# 
# 이와 같은 출력 구조는 후처리 단계에서 sigmoid (binary) 또는 softmax (multi-class) 선택이 자유로워지고,
# 다양한 task에 동일 모델을 재사용할 수 있는 유연성을 제공함.

# 4. 인코더·디코더 구조 단순화: Sequential 블록화
#
# - 기존 코드에서는 각 Conv + BN + ReLU layer를 개별적으로 정의하고, forward()에서 순차적으로 호출함.
# - 이 방식은 layer가 많아질수록 코드가 복잡하고 유지보수가 어려워짐.
#
# → 동일한 CBR 구조를 nn.Sequential로 묶어 블록 단위로 정의함으로써 구조를 명확히 하고 forward() 코드를 간결화함.
# 
# 예: 
#     기존 → self.enc1_1, self.enc1_2 따로 정의
#     변경 → self.enc1 = nn.Sequential(CBR(...), CBR(...))
#
# 이로써 코드의 가독성, 일관성, 유지보수성이 크게 향상되며 구조 자체의 의미 전달이 명확해짐.

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()

        # Conv2d + BatchNorm + ReLU 블록
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            ]
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = nn.Sequential(
            CBR2d(1, 64),
            CBR2d(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            CBR2d(64, 128),
            CBR2d(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            CBR2d(128, 256),
            CBR2d(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            CBR2d(256, 512),
            CBR2d(512, 512),
            nn.Dropout(0.5)  # Dropout 추가
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            CBR2d(512, 1024),
            nn.Dropout(0.5)  # Dropout 추가
        )

        # Decoder (Upsample + Conv 구조 사용)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = nn.Sequential(
            CBR2d(1024 + 512, 512),
            CBR2d(512, 256)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            CBR2d(256 + 256, 256),
            CBR2d(256, 128)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            CBR2d(128 + 128, 128),
            CBR2d(128, 64)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            CBR2d(64 + 64, 64),
            CBR2d(64, 64)
        )

        # Output layer (multi-class 대응 가능)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out