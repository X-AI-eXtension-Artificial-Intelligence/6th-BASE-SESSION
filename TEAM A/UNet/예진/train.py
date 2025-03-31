## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## Parser 생성하기:  코드를 실행할 때 옵션을 입력받을 수 있게 해주는 도구
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")                   # 학습률 
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")        # 배치크기
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")        # 에폭 수 

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir") # 데이터 경로
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir") # 체크포인트 저장 경로 
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")         # 로그 저장 경로 
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir") # 결과 이미지 저장 경로

parser.add_argument("--mode", default="train", type=str, dest="mode")               # train or test 
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue") # 이전 학습 이어서 할 지 

args = parser.parse_args()

## 트레이닝 파라미터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA(GPU)가 가능하면 GPU 사용

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
# 예측 결과를 저장할 폴더가 없으면 새로 만들어줘
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    # 전처리 적용 순서 정의 (정규화 -> 랜덤 뒤집기 -> 텐서 변환)
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform) # Dataset 클래스를 통해 .npy 파일 불러오기
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8) # 배치 단위로 모델에 공급

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform) 
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    # 전체 데이터 개수와 배치 개수 계산
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기 (텐서-> 넘파이, 정규화 해제, 0.5이상은 1 / 미만은 0으로 분류)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정: 학습과정을 시각화할 수 있도록 로그 저장
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":          # 이전에 저장된 모델(checkpoint)을 불러와서 이어서 학습함
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)


   # 학습 루프 (에폭 단위 반복)
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()              # 모델을 학습 모드로 설정
        loss_arr = []            # 손실값 저장 리스트


        for batch, data in enumerate(loader_train, 1):  # 학습 데이터 전체를 작은 배치로 나눠서 순서대로 불러옴
            # forward pass (모델에 입력 넣고 예측)
            label = data['label'].to(device)  # 정답 라벨을 GPU로
            input = data['input'].to(device)  # 입력 이미지도 GPU로

            output = net(input)               # 모델에 입력 넣고 출력 계산

            # backward pass (오차 계산하고 역전파)
            optim.zero_grad()                 # 이전 배치에서의 gradient 초기화

            # 손실 기록 및 출력
            loss = fn_loss(output, label)     # 예측값과 정답 사이의 오차 계산
            loss.backward()                   # 오차를 역전파해서 gradient 계산
            
            optim.step()                      # gradient로 가중치 업데이트


            # 손실함수 계산
            loss_arr += [loss.item()]         # 현재 배치 손실값 저장

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))   # 현재 에폭, 배치, 평균 손실 출력

            # Tensorboard에 이미지 저장하기
            label = fn_tonumpy(label)                         # 라벨을 넘파이로
            input = fn_tonumpy(fn_denorm(input, 0.5, 0.5))    # 입력 정규화 해제 후 넘파이로
            output = fn_tonumpy(fn_class(output))             # 출력 이진화 후 넘파이로

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch) # TensorBoard에 평균 손실 저장

        
        # 검증 단계

        with torch.no_grad():    # gradient 계산 끔 (속도 증가, 메모리 감소)
            net.eval()           # 모델을 평가 모드로 변경
            loss_arr = []        # 손실값 리스트 초기화

            # 검증용 데이터 배치 반복
            for batch, data in enumerate(loader_val, 1):
                # forward pass (예측 + 손실 계산)
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                #결과 출력
                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch) # 평균 검증 손실 저장
        
        # 50 에폭마다 모델 저장 
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE ===========================================================================================
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 저장된 모델 불러오기

    with torch.no_grad(): # 평가 모드 설정
        net.eval()
        loss_arr = []
        # 테스트 데이터 반복
        for batch, data in enumerate(loader_test, 1):
            # forward pass (예측 + 손실 계산)
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기 (이미지 및 넘파이 저장)
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):     # 배치 안 이미지 개수만큼 반복
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))