# 라이브러리 임포트
import os
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AttentionUNet
from dataset import *
from util import *

import matplotlib.pyplot as plt 
from torchvision import transforms, datasets

# 훈련 파라미터 설정
lr = 0.001
batch_size = 4
num_epoch = 100

# 기본 경로, 데이터셋 경로, 모델 체크포인트 경로 지정
base_dir = '/home/work/XAI_BASE/BASE_4주차'
data_dir = '/home/work/XAI_BASE/BASE_4주차/dataset'
ckpt_dir = os.path.join(base_dir, "checkpoint")

os.makedirs(ckpt_dir, exist_ok=True)

# 전처리 Transform 구성 (정규화 -> 랜덤 플립 -> 텐서 변환)
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# DataLoader 구성
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AttentionUNet().to(device)

# 손실함수 정의 (Binary Cross Entropy with Logits)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 설정 (Adam)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 데이터 및 배치 개수 지정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# 네트워크 학습 (에포크 초기화)
st_epoch = 0

# 만약 학습한 모델이 있다면 모델 로드
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# validation loss 최저값 저장을 위한 변수 초기화
best_loss = float('inf')

# 1 에포크부터 num_epoch까지 반복
for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()  # 훈련 모드
    loss_arr = []  # 배치 손실을 저장할 리스트

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # 역전파
        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr.append(loss.item())
        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

    # 검증 과정
    with torch.no_grad():
        net.eval()
        loss_arr = []
        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)
            loss = fn_loss(output, label)
            loss_arr.append(loss.item())
            print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

        # 현재 에포크의 평균 Validation loss 계산
        val_loss = np.mean(loss_arr)
        print("EPOCH %04d | VAL LOSS: %.4f" % (epoch, val_loss))

        # 가장 낮은 Validation loss를 가진 모델 저장
        if val_loss < best_loss:
            best_loss = val_loss
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
            print(">> Best model updated and saved at epoch %d with loss: %.4f" % (epoch, best_loss))
