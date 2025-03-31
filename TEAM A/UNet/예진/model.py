# UNet: 이미지를 입력하면 이미지 안에 무엇이 어디에 있는지 구분해주는 딥러닝 모델
# 이미지 한 장을 넣고, 픽셀 단위로 뭐가 뭔지 구분해서 다시 이미지로 뱉어내야 함!
# 입력 이미지를 -> 작게 줄여가면서 -> 특징을 뽑고 -> 다시 키우면서 원래 크기로 복원하는 구조

import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 구축하기
class UNet(nn.Module):      # PyTorch의 기본 신경망 클래스 상속
    def __init__(self):
        super(UNet, self).__init__()

        # CBR2d: Conv(합성곱) + BN(정규화) + ReLU(활성화)를 묶어서 하나의 블록으로 만들어주는 함수
        # => 이미지를 처리해서 특징을 뽑아주는 조합
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,    # 이미지 처리 핵심 (필터)
                                 kernel_size=kernel_size, stride=stride, padding=padding,  
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]                       # 값 정규화해서 안정적으로 만듦
            layers += [nn.ReLU()]                                                       # 음수 제거, 계산 더 잘되게 만듦

            cbr = nn.Sequential(*layers)  # 3개를 순서대로 묶음

            return cbr

        # 본격적으로 모델 제작

        # 인코더 (이미지 줄이기)
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 1: 입력 이미지가 흑백(채널 1개)이라는 뜻
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64) # 64개의 특징을 뽑아냄

        self.pool1 = nn.MaxPool2d(kernel_size=2)             # 맥스풀링: 이미지를 절반 크기로 줄임

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)  # 2, 3, 4... 반복
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128) # 채널 수는 점점 증가 (128, 256...), 이미지 크기 점점 감소 

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)  # bottleneck: 가장 압축된 지점, 이미지 엄청 줄여서 특징만 남긴 마지막 구간

        # Expansive path
        # 디코더 (이미지 키우기)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)  

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,              # 언풀링: 이미지를 2배로 늘려줌
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)        # 인코더에서 저장해뒀던 512짜리 특징을 붙여서
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)            # 총 1024 채널 → 다시 512로 줄이면서 복원
        
        # 과정 계속 반복 (채널 수 감소, 이미지 크기 증가)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,  
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True) # 최종 채널 1개로 복원해서 출력

    # forward 함수: 입력 x를 받고 내부 거쳐서 예측 결과 반환
    def forward(self, x):               # x: 입력 이미지 (예: 1×256×256 텐서)
        enc1_1 = self.enc1_1(x)         # 여기서 이미지 특징 뽑고 줄이기 반복 (이미지 크기 줄어들고 특징 깊어짐)
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

        enc5_1 = self.enc5_1(pool4)      # 가장 압축된 곳

        dec5_1 = self.dec5_1(enc5_1)     # 이미지 크기 키우기 시작

        unpool4 = self.unpool4(dec5_1)               # 언풀링 -> 이미지 크기 키우기
        cat4 = torch.cat((unpool4, enc4_2), dim=1)   # cat: 인코더에서 저장했던 결과를 같이 써서 더 정확하게 복원함
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

        x = self.fc(dec1_1)   # 최종 예측 결과 반환 (채널 1개짜리 이미지)

        return x