## 필요한 라이브러리 임포트
import argparse  # 명령행 인자를 파싱하기 위한 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet  # U-Net 모델 정의
from dataset import *   # 커스텀 데이터셋 관련 모듈
from util import *      # 학습 관련 유틸 함수 (예: 저장, 불러오기 등)

import matplotlib.pyplot as plt
from torchvision import transforms, datasets

## 명령행 인자 파싱을 위한 parser 설정
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 학습 하이퍼파라미터 및 경로 설정
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")  # train 또는 test
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")  # 학습 이어서 할지 여부

args = parser.parse_args()

## 파라미터 변수에 저장
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# GPU 사용 여부 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 파라미터 출력
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 결과 저장 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 데이터셋 및 데이터로더 구성
if mode == 'train':
    # 데이터 변환 정의 (Normalize, RandomFlip, Tensor 변환)
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    # 학습 및 검증 데이터셋과 로더 생성
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    # 테스트 모드에서는 RandomFlip 없이 Normalize, Tensor 변환만
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    # 테스트 데이터셋 및 로더 생성
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 및 손실함수 정의
net = UNet().to(device)  # U-Net 모델
fn_loss = nn.BCEWithLogitsLoss().to(device)  # 바이너리 크로스엔트로피 손실

## 옵티마이저 정의
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 기타 함수들 정의
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # 텐서를 넘파이로
fn_denorm = lambda x, mean, std: (x * std) + mean  # 역정규화
fn_class = lambda x: 1.0 * (x > 0.5)  # 0.5 기준으로 이진 분류 결과 생성

## 텐서보드 로깅을 위한 SummaryWriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 학습 시작 에폭 설정
st_epoch = 0

# ========================= #
#        TRAIN MODE         #
# ========================= #
if mode == 'train':
    # 이어서 학습하는 경우 체크포인트 불러오기
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # 입력과 라벨 GPU에 올리기
            label = data['label'].to(device)
            input = data['input'].to(device)

            # 순전파
            output = net(input)

            # 역전파
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

            # 학습 상태 출력
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # 텐서보드에 이미지 기록
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # ===================== #
        #     VALIDATION STEP   #
        # ===================== #
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # 주기적으로 체크포인트 저장
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# ==========================#
#        TEST MODE          #
# ========================= #
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            # 이미지 및 넘파이 파일 저장
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
