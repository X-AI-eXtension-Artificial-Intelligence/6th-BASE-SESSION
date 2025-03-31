## 라이브러리 추가하기
import argparse # 파라미터를 입력받기 위한 모듈

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

# Parser 생성하기
## 옵션을 지정할 수 있도록 하는 객체 생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## 학습률, 배치 크기, epoch 수, 데이터, 체크포인트, 로그, 결과 폴더, 실행 모드, 이어서 학습 여부에 대해서 지정하고 있음
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

## 입력받은 인자들 저장
args = parser.parse_args()

# 트레이닝 파라미터 설정하기
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

# 결과 저장할 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# 네트워크 학습하기
if mode == 'train': # train일 때는 RandomFlip 이나 정규화 등이 포함됨
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    ## 데이터 로드
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8) # 검증은 항상 같은 순서로 평가해야 하기 때문에 Shuffle=False 

    # train & val 데이터 개수 계산
    ## 샘플 수 계산
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    ## 배치 수 계산
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else: # test mode
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]) # 테스트용 전처리 -> RandomFlip X

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8) # shuffle=False 

    # 전체 학습 데이터 수 / 배치 수 계산
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

# 네트워크 생성하기
net = UNet().to(device)

# 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device) # 이진 분류용 loss 사용 -> Binary Cross Entropy with Logits의 경우, 출력에 sigmoid가 포함된 BCE 형태라 sigmoid를 따로 쓰지 않아도 됨 

# Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr) # Adam optimizer 사용

# 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) # 텐서를 넘파이 배열로 변환 / GPU를 CPU로 옮기는 이유 : 파이토치 텐서는 GPU에 올라가 있으면 바로 NumPy로 바꿀 수 없음 
fn_denorm = lambda x, mean, std: (x * std) + mean # 정규화 해제
fn_class = lambda x: 1.0 * (x > 0.5) # 0.5 기준으로 이진화

# Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# 네트워크 학습시키기
st_epoch = 0 # 시작 epoch 설정 

# TRAIN MODE
if mode == 'train':
    if train_continue == "on": # 이어서 학습할 것인지 확인
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 저장해둔 거 쓰기

    for epoch in range(st_epoch + 1, num_epoch + 1): # 복원한 st_epoch ~ num_epoch까지 학습 반복
        net.train() # -> 학습 모드로 전환해주면 dropout, batchnorm 등 활성화
        loss_arr = [] # 손실 기록할 리스트 초기화

        for batch, data in enumerate(loader_train, 1): # 훈련 데이터 반복 
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label) # 손실 계산
            loss.backward() # 역전파

            optim.step() # 파라미터 업데이트

            # 손실함수 계산
            loss_arr += [loss.item()] # 현재 배치의 손실을 누적해 평균 손실 출력 

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            ## TensorBoard에 label, input, output 저장 
            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch) # epoch별 평균 손실 기록 
        
        ## 검증 시작
        with torch.no_grad():
            net.eval() # train과 거의 동일하지만, grad 없이, 모델을 eval() 모드로 설정함 
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

        if epoch % 50 == 0: # 50epoch마다 저장하고 writer를 닫음 
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 저장된 모델 불러오기 

    with torch.no_grad(): # -> gradient 계산 비활성화
        net.eval() # test mode 
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
