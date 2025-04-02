## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet #model 불러오기
from dataset import * #dataset불러오기
from util import * # util 불러오기

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--lr", default=1e-3, type=float, dest="lr")  # 학습률 설정, 기본값 1e-3
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")  # 배치 크기 설정, 기본값 4
parser.add_argument("--num_epoch", default=5, type=int, dest="num_epoch")  # 학습 에포크 수 설정, 기본값 5

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")  # 데이터 디렉토리 경로 설정
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")  # 체크포인트 저장 디렉토리 설정
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")  # 로그 저장 디렉토리 설정
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")  # 결과 저장 디렉토리 설정

parser.add_argument("--mode", default="train", type=str, dest="mode")  # 실행 모드 설정 (기본값: 'train')
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")  # 학습 연속 여부 설정

# 파서에서 인수 읽기
args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = args.lr  # 학습률
batch_size = args.batch_size  # 배치 크기
num_epoch = args.num_epoch  # 학습 에포크 수

data_dir = args.data_dir  # 데이터 디렉토리
ckpt_dir = args.ckpt_dir  # 체크포인트 디렉토리
log_dir = args.log_dir  # 로그 디렉토리
result_dir = args.result_dir  # 결과 디렉토리

mode = args.mode
train_continue = args.train_continue # 학습 연속 여부

#GPU사용 여부
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 설정된 파라메터 출력
print("learning rate: %.4e" % lr)  # 학습률 출력
print("batch size: %d" % batch_size)  # 배치 크기 출력
print("number of epoch: %d" % num_epoch)  # 학습 에포크 수 출력
print("data dir: %s" % data_dir)  # 데이터 디렉토리 출력
print("ckpt dir: %s" % ckpt_dir)  # 체크포인트 디렉토리 출력
print("log dir: %s" % log_dir)  # 로그 디렉토리 출력
print("result dir: %s" % result_dir)  # 결과 디렉토리 출력
print("mode: %s" % mode)  # 실행 모드 출력

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))  # PNG 파일을 저장할 디렉토리 생성
    os.makedirs(os.path.join(result_dir, 'numpy'))  # NumPy 파일을 저장할 디렉토리 생성

## 네트워크 학습하기
if mode == 'train':
    #데이터 학습 설정정
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    #학습 데이터 셋 불러오기
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    #검증 데이터셋 불러오기
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    #학습/검증 데이터 셋 배치 수수
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    #실행모드가 test인 경우
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)  # 테스트 데이터셋
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)  # 테스트 데이터 로더

    # 데이터셋 크기 및 배치 수 설정
    num_data_test = len(dataset_test)  # 테스트 데이터 개수
    num_batch_test = np.ceil(num_data_test / batch_size)  # 테스트 데이터 배치 수

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr) #Adam사용

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':  # 학습 모드일 경우
    if train_continue == "on":  # 이전 학습을 이어서 하려면
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 이전 학습 체크포인트 로드

    for epoch in range(st_epoch + 1, num_epoch + 1):  # 에포크 수만큼 반복
        net.train()  # 모델을 학습 모드로 설정
        loss_arr = []  # 손실값을 저장할 리스트 초기화

        for batch, data in enumerate(loader_train, 1):  # 배치 단위로 학습
            # forward pass
            label = data['label'].to(device)  # 라벨 데이터를 device로 이동
            input = data['input'].to(device)  # 입력 데이터를 device로 이동

            output = net(input)  # 네트워크에 입력 데이터를 넣어 출력 생성

            # backward pass
            optim.zero_grad()  # 기울기 초기화

            loss = fn_loss(output, label)  # 손실값 계산
            loss.backward()  # 기울기 계산

            optim.step()  # 가중치 업데이트

            # 손실값 저장
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            label = fn_tonumpy(label)  # 라벨을 NumPy 배열로 변환
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))  # 입력 이미지를 정규화 후 NumPy 배열로 변환
            output = fn_tonumpy(fn_class(output))  # 출력 이미지를 클래스별로 분류 후 NumPy 배열로 변환

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)  # Tensorboard에 손실값 기록

        with torch.no_grad():  # 검증 데이터셋에 대해서는 기울기 계산을 하지 않음
            net.eval()  # 모델을 평가 모드로 설정
            loss_arr = []  # 손실값을 저장할 리스트 초기화

            for batch, data in enumerate(loader_val, 1):  # 배치 단위로 검증
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실값 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))  # 검증 진행 

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        #50마다 체크포인트에 저장장
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 학습된 모델과 옵티마이저 로드

    with torch.no_grad():  # 테스트 데이터셋에 대해서는 기울기 계산을 하지 않음
        net.eval()  # 모델을 평가 모드로 설정
        loss_arr = []  # 손실값을 저장할 리스트 초기화

        for batch, data in enumerate(loader_test, 1):  # 배치 단위로 테스트
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실값 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %  # 테스트 진행 상황 출력
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard에 이미지 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):  # 각 배치마다 이미지 저장
                id = num_batch_test * (batch - 1) + j  # 이미지 고유 ID 생성

                # 결과 이미지 저장
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # 결과 데이터를 NumPy 배열로 저장
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %  # 전체 테스트 결과 출력
          (batch, num_batch_test, np.mean(loss_arr)))
