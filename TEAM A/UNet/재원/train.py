# 라이브러리
import os
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt 
from torchvision import transforms, datasets

# 훈련 파라미터 설정
lr = 1e-3
batch_size = 4
num_epoch = 100

# 기본 경로, 데이터셋 경로, 모델 체크포인트 경로 지정
base_dir = '/home/work/XAI_WinterStudy/U-Net'
data_dir = '/home/work/XAI_WinterStudy/U-Net/dataset'
ckpt_dir = os.path.join(base_dir, "checkpoint")

os.makedirs(ckpt_dir, exist_ok=True)

# 전처리 Transform 구성
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# DataLoader 구성
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet().to(device)

# 손실함수 정의하기(Binary CE에 시그모이드 결합된 손실함수)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 설정하기(Adam)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 데이터 개수 지정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

# 배치 개수 지정
num_batch_train = np.ceil(num_data_train / batch_size) #np.ceil -> 나누고 남은 데이터는 유동적으로 마지막 배치에 배정
num_batch_val = np.ceil(num_data_val / batch_size)

# 네트워크 학습(에포크 초기화)
st_epoch = 0

# 만약 학습한 모델이 있다면 모델 로드
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) 

# 1에포크 부터 에포크 개수 만큼 반복
for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train() #훈련 모드
        loss_arr = [] #에포크마다 배치 손실 담을 list


        for batch, data in enumerate(loader_train, 1):
            
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 역전파 
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 계산된 손실 추가
            loss_arr += [loss.item()]

            #학습 로그 출력(배치의 loss 평균)
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))


        #검증 과정
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                
                # 순전파
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                # 학습 로그 출력
                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))


        # epoch 2마다 모델 저장
        if epoch % 2 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

