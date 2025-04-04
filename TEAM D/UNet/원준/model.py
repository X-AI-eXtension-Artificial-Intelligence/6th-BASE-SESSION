import torch
import torch.nn as nn

# Conv2d → BatchNorm2d → ReLU 블록
def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),  # 배치 정규화
        nn.ReLU(inplace=True)
    )

# U-Net 아키텍처 정의
class UNet(nn.Module):
    def __init__(self, depth=3):
        super(UNet, self).__init__()
        self.depth = depth

        self.enc_blocks = nn.ModuleList()    # 인코더 블록 저장 리스트
        self.pool_layers = nn.ModuleList()   # MaxPooling 레이어 저장 리스트

        # 각 인코딩 단계의 채널 수 계산
        chs = [64 * (2 ** i) for i in range(depth)]
        prev_ch = 1  # 입력 채널 수 (예: 흑백이면 1, RGB면 3)

        # 인코더 구성: Conv-BN-ReLU × 2 + MaxPool
        for ch in chs:
            self.enc_blocks.append(nn.Sequential(
                CBR2d(prev_ch, ch),
                CBR2d(ch, ch)
            ))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
            prev_ch = ch

        # Bottleneck: 가장 깊은 단계
        self.bottleneck = nn.Sequential(
            CBR2d(chs[-1], chs[-1] * 2),
            CBR2d(chs[-1] * 2, chs[-1] * 2)
        )

        # 디코더 구성: 업샘플 + skip connection + Conv-BN-ReLU × 2
        self.upconv_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for ch in reversed(chs):
            self.upconv_layers.append(
                nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(nn.Sequential(
                CBR2d(ch * 2, ch),
                CBR2d(ch, ch)
            ))

        # 마지막 출력 Conv: 클래스 수만큼 출력 채널 (여기선 1개로 고정)
        self.final_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        enc_feats = []

        # 인코더: 특징 추출 + 다운샘플링
        for enc_block, pool in zip(self.enc_blocks, self.pool_layers):
            x = enc_block(x)
            enc_feats.append(x)  # skip connection용 저장
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # 디코더: 업샘플링 + skip connection + 특징 합치기
        for upconv, dec_block, enc_feat in zip(self.upconv_layers, self.dec_blocks, reversed(enc_feats)):
            x = upconv(x)

            # 인코딩된 feature와 크기가 안 맞을 경우 보간
            if x.shape[-2:] != enc_feat.shape[-2:]:
                x = nn.functional.interpolate(x, size=enc_feat.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, enc_feat], dim=1)  # skip connection
            x = dec_block(x)

        # 최종 출력
        return self.final_conv(x)



# 원래 모델 

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



# Encoder Stage 1 → Pool1  ←── Depth Level 1
# Encoder Stage 2 → Pool2  ←── Depth Level 2
# Encoder Stage 3 → Pool3  ←── Depth Level 3
# Encoder Stage 4 → Pool4  ←── Depth Level 4
# Encoder Stage 5         ←── Depth Level 5 (bottom, no pooling after this)



# 깊이 조절 가능 코드 

# import torch
# import torch.nn as nn

# class UNet(nn.Module):
#     def __init__(self, depth=3):  # depth 조절 가능
#         super(UNet, self).__init__()

#         self.depth = depth

#         def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )

#         self.enc_blocks = nn.ModuleList()
#         self.pool_layers = nn.ModuleList()

#         chs = [64 * (2 ** i) for i in range(depth)]

#         prev_ch = 1
#         for ch in chs:
#             self.enc_blocks.append(nn.Sequential(CBR2d(prev_ch, ch), CBR2d(ch, ch)))
#             self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
#             prev_ch = ch

#         self.bottleneck = nn.Sequential(
#             CBR2d(chs[-1], chs[-1] * 2),
#             CBR2d(chs[-1] * 2, chs[-1] * 2)
#         )

#         self.upconv_layers = nn.ModuleList()
#         self.dec_blocks = nn.ModuleList()

#         for ch in reversed(chs):
#             self.upconv_layers.append(
#                 nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
#             )
#             self.dec_blocks.append(
#                 nn.Sequential(CBR2d(ch * 2, ch), CBR2d(ch, ch))
#             )

#         self.final_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

#     def forward(self, x):
#         enc_feats = []

#         for enc_block, pool in zip(self.enc_blocks, self.pool_layers):
#             x = enc_block(x)
#             enc_feats.append(x)
#             x = pool(x)

#         x = self.bottleneck(x)

#         for upconv, dec_block, enc_feat in zip(self.upconv_layers, self.dec_blocks, reversed(enc_feats)):
#             x = upconv(x)
#             x = torch.cat([x, enc_feat], dim=1)
#             x = dec_block(x)

#         return self.final_conv(x)

# 예시로 더 얕은 U-Net 생성
# model = UNet(depth=3)
# 예시로 더 깊은 U-Net 생성
# model = UNet(depth=5)


# depth=3 → 얕은 네트워크

# depth=5 이상 → 깊은 네트워크


