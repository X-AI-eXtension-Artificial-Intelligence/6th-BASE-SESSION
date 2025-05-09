import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from dataset import MRIDataset  # ← 새로 만든 Dataset
from util import *  # load, save 함수 등
import matplotlib.pyplot as plt

## 파라미터 입력
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_epoch", type=int, default=150)

parser.add_argument("--data_dir", type=str, default="./datasets")
parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
parser.add_argument("--log_dir", type=str, default="./log")
parser.add_argument("--result_dir", type=str, default="./result")

parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--train_continue", type=str, default="off")
args = parser.parse_args()

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 결과 디렉토리 생성
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(args.result_dir, 'numpy'), exist_ok=True)

# TensorBoard 로그 저장소
writer_train = SummaryWriter(os.path.join(args.log_dir, 'train'))
writer_val = SummaryWriter(os.path.join(args.log_dir, 'val'))

# 데이터셋 불러오기
if args.mode == 'train':
    dataset_train = MRIDataset(data_dir=os.path.join(args.data_dir, 'train'))
    dataset_val = MRIDataset(data_dir=os.path.join(args.data_dir, 'val'))

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

else:
    dataset_test = MRIDataset(data_dir=os.path.join(args.data_dir, 'test'))
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

# 모델 정의
net = UNet().to(device)

# 손실함수 및 옵티마이저
fn_loss = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# 후처리 함수
fn_tonumpy = lambda x: x.detach().cpu().numpy().transpose(0, 2, 3, 1)
fn_class = lambda x: (x > 0.5).float()

# 학습 이어서 하기
start_epoch = 0
if args.train_continue == 'on':
    net, optimizer, start_epoch = load(args.ckpt_dir, net, optimizer)

# 학습 루프
if args.mode == 'train':
    for epoch in range(start_epoch + 1, args.num_epoch + 1):
        net.train()
        train_losses = []

        for batch, data in enumerate(loader_train):
            inputs = data['input'].to(device)
            labels = data['label'].to(device)

            outputs = net(inputs)
            loss = fn_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print(f"[Epoch {epoch:03d}] TRAIN LOSS: {np.mean(train_losses):.4f}")
        writer_train.add_scalar('loss', np.mean(train_losses), epoch)

        # 검증
        net.eval()
        val_losses = []
        with torch.no_grad():
            for data in loader_val:
                inputs = data['input'].to(device)
                labels = data['label'].to(device)

                outputs = net(inputs)
                loss = fn_loss(outputs, labels)
                val_losses.append(loss.item())

        print(f"[Epoch {epoch:03d}] VAL LOSS: {np.mean(val_losses):.4f}")
        writer_val.add_scalar('loss', np.mean(val_losses), epoch)

        save(args.ckpt_dir, net, optimizer, epoch)

    writer_train.close()
    writer_val.close()

# 테스트 루프
else:
    net.eval()
    net, optimizer, _ = load(args.ckpt_dir, net, optimizer)
    test_losses = []
    with torch.no_grad():
        for batch, data in enumerate(loader_test):
            inputs = data['input'].to(device)
            labels = data['label'].to(device)

            outputs = net(inputs)
            loss = fn_loss(outputs, labels)
            test_losses.append(loss.item())

            # 결과 저장
            inputs_np = fn_tonumpy(inputs)
            labels_np = fn_tonumpy(labels)
            outputs_np = fn_tonumpy(fn_class(outputs))

            for i in range(inputs_np.shape[0]):
                save_image(os.path.join(args.result_dir, 'png', f'{batch:03d}_{i}_input.png'), inputs_np[i])
                save_image(os.path.join(args.result_dir, 'png', f'{batch:03d}_{i}_label.png'), labels_np[i])
                save_image(os.path.join(args.result_dir, 'png', f'{batch:03d}_{i}_output.png'), outputs_np[i])

    print(f"AVERAGE TEST LOSS: {np.mean(test_losses):.4f}")
