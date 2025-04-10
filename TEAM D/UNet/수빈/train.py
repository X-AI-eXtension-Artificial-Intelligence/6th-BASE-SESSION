# 라이브러리 추가
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import UNet  # 기존 U-Net 대신 경량화된 U-Net 사용
from dataset import *
from util import *
import argparse
import matplotlib.pyplot as plt

# Parser 생성(코드 사용을 좀 더 유연하게 할 수 있음)
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 학습률/배치 사이즈/epoch 설정
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# 데이터/체크 포인트/로그 디렉토리 설정
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

# 실행 모드 설정
parser.add_argument("--mode", default="train", type=str, dest="mode")

# 이어서 학습할지 여부 설정
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# args에 파싱해서 저장
args = parser.parse_args()

# 트레이닝 파라메터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

# 디렉토리 설정
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir
mode = args.mode
train_continue = args.train_continue

# 동작 방식(GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 디렉토리 없으면 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# 데이터 변환 (Normalization 및 ToTensor 적용)
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)
    num_batch_train = int(np.ceil(num_data_train / batch_size))
    num_batch_val = int(np.ceil(num_data_val / batch_size))

else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

# 경량화된 U-Net 모델 적용
net = UNet().to(device)  # 기존 U-Net을 불러오지 않고, 수정된 경량화 U-Net 불러오기

# 손실함수 적용 (BCEWithLogitsLoss)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Adam 옵티마이저 적용
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Tensor 변환 관련 함수 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# Tensorboard 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# 학습 과정 반영
st_epoch = 0
if train_continue == "on":
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

if mode == 'train':
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr.append(loss.item())

            print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {int(num_batch_train):04d} | LOSS {np.mean(loss_arr):.4f}")

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # Validation 과정
        with torch.no_grad():
            net.eval()
            loss_arr = []
            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)
                loss_arr.append(loss.item())

                print(f"VALID: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else:
    with torch.no_grad():
        net.eval()
        loss_arr = []
        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)
            loss_arr.append(loss.item())

            print(f"TEST: BATCH {batch:04d} / {num_batch_test:04d} | LOSS {np.mean(loss_arr):.4f}")

    print(f"AVERAGE TEST LOSS: {np.mean(loss_arr):.4f}")
