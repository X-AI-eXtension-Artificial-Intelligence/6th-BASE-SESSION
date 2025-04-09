import os
import numpy as np

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = [] 
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # 성능 향상을 위해 Batchnorm -> Groupnorm으로 교체 (group : 8)
            ## Batchnorm : 배치 크기에 의존함 -> 작은 배치일수록 통계치가 불안정해짐
            ## Groupnorm : 채널 내 그룹 기준 정규화 -> 배치 크기와 상관없이 항상 안정적임
            ## U-Net : 메모리 제한으로 작은 배치를 쓴 구조 -> Groupnorm이 더 안정적이고 성능 향상이 가능함
            layers += [nn.GroupNorm(num_groups=8, num_channels=out_channels)] 
            # 성능 향상을 위해 ReLU -> LeakyReLU로 교체
            ## 죽은 뉴런 문제 해결을 위해서 
            layers += [nn.LeakyReLU(negative_slope=0.01)] 

            cbr = nn.Sequential(*layers) 

            return cbr

        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)


        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # 성능 향상을 위해 Dropout 추가
        ## nn.Dropout2d : Dropout의 CNN 전용 버전으로, 채널 단위로 비활성화 시켜버림 -> 더 강력한 regularization 효과를 가짐 / p=0.3 : 30%의 확률로 비활성화 시킴
        ## 이 자리에 넣은 이유 : 이미지의 가장 압축된 정보가 있기 때문에 Dropout을 통해 모델이 다양한 상황에서도 예측할 수 있도록 학습함 
        self.dropout = nn.Dropout2d(p=0.3)

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

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
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # 성능 향상을 위해 가중치 초기화 추가
        ## Pytorch가 기본적으로 가중치를 자동으로 초기화하기는 하지만, 좋은 초기화 방법을 적용해주면 수렴 속도가 빨라지고, 학습 초반이 안정적이고, 성능도 좋아질 수 있음
        self.apply(self._init_weights) 

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight) # ReLU 계열 활성화 함수와 궁합이 좋은 초기화 방법 (He 초기화의 일종) -> 초깃값의 분산을 잘 조정해서 기울기 소실이나 폭주 없이 학습할 수 있도록 함
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # 초기 편향이 없도록, 즉 중립적으로 시작할 수 있도록 0으로 설정 (학습을 통해 최적값으로 조정될 수 있기 때문에 랜덤하게 설정할 필요가 없음)

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

        # 성능 향상을 위해 Dropdout 추가
        enc5_1 = self.dropout(enc5_1)

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