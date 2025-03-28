## 라이브러리 추가하기
import argparse # 커맨드 인자 파싱 이 값들을 파이썬 코드 안에서 받아와서 쓰기 위해 사용하는 게 바로 argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet # UNet 모델 정의 모듈
from dataset import *  # Dataset, Transform 정의한 모듈
from util import *     # save, load 함수 등 유틸리티 모음

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter) # 기본값도 함께 출력되도록 포맷을 설정

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")                         # 학습률(learning rate)을 외부에서 설정
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")              # 미니배치 크기 설정정
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")              # 에폭 수 설정

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")       # 데이터셋이 저장된 폴더 경로
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")     # 체크포인트
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")              # TensorBoard 로그 파일이 저장될 경로
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")     # 예측 결과 이미지나 .npy 파일을 저장할 폴더

# 실행 모드 선택
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")  # 저장된 모델 불러와서 이어서 학습함

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])  # [-1, 1] 범위로 정규화 , 데이터 증강, 텐서(C, H, W)로 변환 

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)  # 학습 시 데이터를 섞어서 모델 일반화 향상, 데이터를 병렬로 빠르게 로딩

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)     # 섞지 않고 순서대로 사용해야 평가 결과가 일정

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)                                        # 배치 수는 소수점 반올림 없이 올림 처리 (마지막 배치가 꽉 차지 않더라도 포함)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])                # 일관된 입력이 중요하므로 증강 안 함

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)  
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)   # 예측 결과를 저장하거나 비교할 때 순서가 중요하므로 섞지 않음

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
## BCEWithLogitsLoss = sigmoid + binary cross entropy 조합으로 이진분류에 많이 사용
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # NumPy 배열 형식으로 바꿔주는 유틸 함수 (텐서를 cup로 그레디언트 삭제, 넘파이로 바꾸기 , 텐서차원으로 변경)
fn_denorm = lambda x, mean, std: (x * std) + mean # 정규화 복원원          # PyTorch 텐서 형식은 (B, C, H, W)
fn_class = lambda x: 1.0 * (x > 0.5) # 임계값 0.5

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()               # 이전의 gradient 초기화

            loss = fn_loss(output, label)   # 손실 계산 (BCEWithLogitsLoss)
            loss.backward()                 # gradient 계산

            optim.step()                    # 파라미터 업데이트

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %    # 손실함수 출력력
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()      # 모델을 평가 모드로 설정
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):  # 배치 검증 반복
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # 50 에폭마다 모델을 저장해두기
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    # TensorBoard writer 종료
    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():       # gradient 계산을 끔
        net.eval()      
        loss_arr = []

        # 테스트 데이터셋을 한 배치씩 불러오면서 평가
        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]       # 손실값을 리스트에 계속 추가
            
            # 진행상황 출력
            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))     # 정규화 복원
            output = fn_tonumpy(fn_class(output))                       # 이진화

            # 결과 파일 저장
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # label, input, output 이미지를 각각 .png로 저장 (흑백)
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # 같은 데이터를 .npy로도 저장 (후처리, 분석용)
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
                
    # 최종 평균 손실 출력
    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))

