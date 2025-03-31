## 라이브러리 추가하기
import argparse  # 명령줄 인자(argument) 파싱을 도와주는 파이썬 내장 모듈
                 # 코드를 실행할 때 외부에서 값을 입력할 수 있게

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # PyTorch에서 TensorBoard 로그를 기록하기 위한 도구
                                                   # 학습 중 손실, 정확도, 이미지 등 다양한 정보를 시각화 가능 

from model import UNet  
from dataset import * 
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets


""" Parser 생성하기. 
파라미터를 코드 안에서 직접 수정하지 않고 터미널에서 --옵션명 값 형식으로 넘겨줄 수 있도록 
"""
parser = argparse.ArgumentParser(description="Train the UNet",  # 도움말 출력 시 보여줄 문구
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 기본값도 도움말에 자동으로 보여줘서 더 친절함

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")  # 학습률 기본값 0.001 
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")  # 배치사이즈 기본값 4
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")  # 에폭 기본값 100 

# 데이터셋, 체크포인트 저장, 로그 저장, 결과 이미지 저장 경로를 설정
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

# mode: "train" 또는 "test" 등 실행 목적
# train_continue: "on"이면 저장된 모델에서 이어서 학습
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# 명령줄에서 넘긴 인자들을 받아서 args.lr, args.batch_size 등으로 접근 가능하게 만듦
args = parser.parse_args()


"""트레이닝 파라미터 설정하기
 명령줄에서 입력받은 학습 파라미터를 실제 변수로 저장하고
 현재 설정을 출력해서 실험 정보를 확인"""
lr = args.lr  # 학습률
batch_size = args.batch_size  # 배치 
num_epoch = args.num_epoch  # 에폭  

data_dir = args.data_dir  # 데이터 경로 
ckpt_dir = args.ckpt_dir  # 체크포인트 
log_dir = args.log_dir  # 로그 
result_dir = args.result_dir  # 결과 들의 저장 경로 

mode = args.mode  # train, test 모드 중 하나 
train_continue = args.train_continue  # on 이면 이어서 학습 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 현재 세팅값들을 출력해서 터미널에 로그로. 실험 재현(reproducibility)을 위해 매우 유용!
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
if mode == 'train':  # 학습모드이면  
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])  # transform정의 

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)  # train 데이터 처리 
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)  # 데이터 로더에 싣기

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)  # val 데이터 처리 
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)  # 3 데이터 로더에 

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)  # 훈련 데이터 수 
    num_data_val = len(dataset_val)  # 검증 데이터 수 
    num_batch_train = np.ceil(num_data_train / batch_size)  # 훈련셋 배치 적용 수 
    num_batch_val = np.ceil(num_data_val / batch_size)  # 검증셋 배치 적용 수 

else:  # 평가모드이면 
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)  # 테스트 셋 수 

    num_batch_test = np.ceil(num_data_test / batch_size)  # 테스트 셋 배치 적용 수 

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)  # Adam, 학습률 

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # 텐서를 넘파이 배열로. (N, C, H, W) → (N, H, W, C)로 바꿔서 이미지처럼 보이게
fn_denorm = lambda x, mean, std: (x * std) + mean  # 정규화 했던 거 복원 
fn_class = lambda x: 1.0 * (x > 0.5)  # 0.5보다 크면 1, 작으면 0 

## Tensorboard 를 사용하기 위한 SummaryWriter설정(TensorBoard에서 로그를 기록하기 위한 객체)
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))  # train폴더에 저장 
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))  # val 폴더에 저장 

## 네트워크 학습시키기
st_epoch = 0

'''학습 + 검증 과정을 반복하면서 TensorBoard에 로그를 저장하고, 지정된 에폭마다 모델을 저장하는 핵심 코드''' 
if mode == 'train': # TRAIN MODE
    if train_continue == "on":  # 저장된 모델이면 이어서 학습 
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):  # 저장된 모델이면 에폭에서 이어서 학습 
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)  # 모델 통과 


            optim.zero_grad()  # 옵티마이저 초기화 
            loss = fn_loss(output, label)  # 손실함수값 
            loss.backward()  # 역전파 
            optim.step()  # 가중치 갱신 
            loss_arr += [loss.item()] # 손실함수 계산

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %  # 실시간 로그 출력 
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)  # label을 넘파이로 
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))  # input을 정규화 해제, 넘파이로 
            output = fn_tonumpy(fn_class(output))  # 결과를 0 또는 1로 바꿔서 넘파이로   

            writer_train.add_image('label',  # tensor보드에 보일 이름 
                                    label,   # 저장할 값 
                                    num_batch_train * (epoch - 1) + batch,  # global step. 5번째 epoch의 3번째 batch라면 4 × num_batch + 3
                                    dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():  # 역전파, 가중치 갱신 x -> 평가모드 
            net.eval()  # BN 비활성화 
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
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

        if epoch % 50 == 0:  # 50번 주기로 네트워크 저장 
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)  # util.py에서 정의된 함수 

    writer_train.close()
    writer_val.close()


else:  # TEST MODE
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # load()는 학습 중 저장한 모델 파라미터 및 optimizer 상태를 복원

    with torch.no_grad():  # 평가모드 
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))

