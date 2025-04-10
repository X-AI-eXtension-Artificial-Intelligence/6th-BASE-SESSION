## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torchvision import transforms

from model import UNet
from dataset import Dataset, Normalization, ToTensor
from util import load

## 트레이닝 파라메터 설정하기
lr = 1e-3                 # 학습률 설정
batch_size = 4            # 배치 사이즈 설정
num_epoch = 100           # 최대 에폭 수

## 디렉토리 경로 설정
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './results'

# 결과 저장 폴더가 없으면 생성
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 데이터 로더 설정
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

## 네트워크 및 손실함수 정의
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 유틸 함수 정의
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## 네트워크 불러오기
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

## 테스트 시작
with torch.no_grad():
    net.eval()
    loss_arr = []

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        # 결과 저장
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = int(num_batch_test * (batch - 1) + j)

            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output[j].squeeze())

            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))