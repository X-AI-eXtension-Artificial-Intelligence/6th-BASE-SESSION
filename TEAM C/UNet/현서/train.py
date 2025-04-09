import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from model import UNet
from dataset import *

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Dataset, ToTensor 
from utils import save, load
from utils import generate_dummy_data


# 훈련 파라미터 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 300


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'data'
ckpt_dir = 'checkpoint'
result_dir = 'result'
log_dir = 'log'


# data_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/data'
# ckpt_dir ='/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/checkpoint'
# result_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/result'
# log_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/log'


# 디렉토리가 없으면 생성
os.makedirs(data_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
print('파일 넘어감')

mode = 'train'  # 기본값 세팅 (나중에 test 모드로도 변경 가능)

# Transform 정의
transform = ToTensor()

# Dataset 인스턴스
os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)

# 경로 설정 이후에 호출 (보통 dataset 정의 전에!)
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# generate_dummy_data(train_dir, num_samples=5)  # train용 더미 데이터 생성
# generate_dummy_data(val_dir, num_samples=2)    # val용 더미 데이터 생성


# DataLoader 정의
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)


print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

# 네트워크 생성
net = UNet().to(device)

# 손실함수 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 생성
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적인 variables
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)


# output을 저장하기 위한 함수
# tensor to numpy
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# norm to denorm
fn_denorm = lambda x, mean, std: (x * std) + mean
# network output을 binary로 분류
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

net, optim, st_epoch = load(ckpt_dir = ckpt_dir, net =net, optim =optim)

# 네트워크 학습
st_epoch = 0
for epoch in range(st_epoch + 1, num_epoch +1):
  net.train()
  loss_arr = []

  for batch, data in enumerate(loader_train, 1):
    # forward pass
    label = data['label'].to(device)
    input = data['input'].to(device)
    output = net(input)
    

    # backward pass
    optim.zero_grad()
    loss = fn_loss(output, label)
    loss.backward()

    optim.step()

    # 손실함수 계산
    loss_arr +=[loss.item()]
    print("Train : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f"% (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
    
    # 텐서보드에 저장
    label = fn_tonumpy(label)
    input= fn_tonumpy(fn_denorm(input, mean = 0.5, std =0.5))
    output = fn_tonumpy(fn_class(output))

    
    writer_train.add_image('label', label, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
    writer_train.add_image('input', input, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
    writer_train.add_image('output', output, num_batch_train * (epoch -1) + batch, dataformats ='NHWC')
    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)


  # validation
  with torch.no_grad():
    net.eval()
    loss_arr = []
    for batch, data in enumerate(loader_val, 1):

      # forward pass
      label = data['label'].to(device)
      input = data['input'].to(device)

      output = net(input)

      # 손실함수 계산하기
      loss = fn_loss(output, label)
      loss_arr +=[loss.item()]
      print("Train : Epoch %04d / %04d | Batch %04d / %04d | Loss %.4f"% (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
      
      # Tensorboard 저장하기
      label = fn_tonumpy(label)
      input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
      output = fn_tonumpy(fn_class(output))

      writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
      writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
      writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

  writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

  # epoch 50 마다 모델 저장하기
  if epoch % 50 == 0:
    save(ckpt_dir = ckpt_dir, net = net, optim= optim, epoch = epoch)
    
writer_train.close()
writer_val.close()
     
