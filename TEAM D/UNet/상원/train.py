# train.py
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import AttentionUNet, DiceLoss
from dataset import Dataset, Normalization, RandomFlip, ToTensor
from util import save, load, fn_tonumpy, fn_denorm, fn_class
from dataset import RandomRotate, AddNoise

# argparse 설정
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--data_dir', type=str, default='./datasets')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--result_dir', type=str, default='./results')
parser.add_argument('--train_continue', type=str, default='off')
args = parser.parse_args()

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 디렉토리 생성
if not os.path.exists(args.result_dir):
    os.makedirs(os.path.join(args.result_dir, 'png'))
    os.makedirs(os.path.join(args.result_dir, 'numpy'))

# 데이터셋 및 DataLoader
transform = transforms.Compose([
    RandomFlip(),
    RandomRotate(),
    AddNoise(),
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

train_dataset = Dataset(data_dir=os.path.join(args.data_dir, 'train'), transform=transform)
val_dataset = Dataset(data_dir=os.path.join(args.data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# 네트워크, 손실함수, 옵티마이저
net = AttentionUNet().to(device)
dice = DiceLoss().to(device)
bce = nn.BCEWithLogitsLoss().to(device)
fn_loss = lambda output, label: 0.5 * dice(output, label) + 0.5 * bce(output, label)

optim = torch.optim.Adam(net.parameters(), lr=args.lr)

# TensorBoard 설정
writer_train = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(args.log_dir, 'val'))

# 이어서 학습 시 로드
st_epoch = 0
if args.train_continue == 'on':
    net, optim, st_epoch = load(args.ckpt_dir, net, optim)

# 학습 루프
for epoch in range(st_epoch + 1, args.num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(train_loader, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr += [loss.item()]

        print(f"TRAIN: EPOCH {epoch:04d} | BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")

        # Tensorboard 저장
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        label_np = fn_tonumpy(label)
        output_np = fn_tonumpy(fn_class(output))

        writer_train.add_image('input', input_np, epoch * batch, dataformats='NHWC')
        writer_train.add_image('label', label_np, epoch * batch, dataformats='NHWC')
        writer_train.add_image('output', output_np, epoch * batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # 검증 루프
    net.eval()
    loss_arr = []
    with torch.no_grad():
        for batch, data in enumerate(val_loader, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print(f"VALID: EPOCH {epoch:04d} | BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")

            input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
            label_np = fn_tonumpy(label)
            output_np = fn_tonumpy(fn_class(output))

            writer_val.add_image('input', input_np, epoch * batch, dataformats='NHWC')
            writer_val.add_image('label', label_np, epoch * batch, dataformats='NHWC')
            writer_val.add_image('output', output_np, epoch * batch, dataformats='NHWC')

    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    # 주기적 저장
    if epoch % 50 == 0:
        save(args.ckpt_dir, net, optim, epoch)

writer_train.close()
writer_val.close()