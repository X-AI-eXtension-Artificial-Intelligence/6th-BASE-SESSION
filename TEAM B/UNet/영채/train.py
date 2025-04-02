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
from torchvision import transforms

# 하이퍼파라미터 설정
parser = argparse.ArgumentParser(description="Train the UNet")
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_epoch", default=100, type=int)
parser.add_argument("--data_dir", default="./datasets", type=str)
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str)
parser.add_argument("--log_dir", default="./log", type=str)
parser.add_argument("--result_dir", default="./result", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--train_continue", default="off", type=str)
args = parser.parse_args()

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

print(f"learning rate: {lr:.4e}")
print(f"batch size: {batch_size}")
print(f"number of epoch: {num_epoch}")
print(f"data dir: {data_dir}")
print(f"ckpt dir: {ckpt_dir}")
print(f"log dir: {log_dir}")
print(f"result dir: {result_dir}")
print(f"mode: {mode}")


os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# DataLoader 설정
if mode == 'train':
    transform = transforms.Compose([Normalization(0.5, 0.5), RandomFlip(), ToTensor()])
    dataset_train = Dataset(os.path.join(data_dir, 'train'), transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dataset_val = Dataset(os.path.join(data_dir, 'val'), transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)
    num_batch_train = np.ceil(len(dataset_train) / batch_size)
    num_batch_val = np.ceil(len(dataset_val) / batch_size)
else:
    transform = transforms.Compose([Normalization(0.5, 0.5), ToTensor()])
    dataset_test = Dataset(os.path.join(data_dir, 'test'), transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    num_batch_test = np.ceil(len(dataset_test) / batch_size)

# 모델, 손실함수, 옵티마이저 정의
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# 텐서보드 설정
writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(os.path.join(log_dir, 'val'))

# 학습
st_epoch = 0

if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir, net, optim)

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

            print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | "
                  f"BATCH {batch:04d} / {int(num_batch_train):04d} | LOSS {np.mean(loss_arr):.4f}")

            # Tensorboard 
            label_np = fn_tonumpy(label)
            input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
            output_np = fn_tonumpy(fn_class(output))
            step = int(num_batch_train) * (epoch - 1) + batch
            writer_train.add_image('label', label_np, step, dataformats='NHWC')
            writer_train.add_image('input', input_np, step, dataformats='NHWC')
            writer_train.add_image('output', output_np, step, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)
                output = net(input)
                loss = fn_loss(output, label)
                loss_arr.append(loss.item())

                print(f"VALID: EPOCH {epoch:04d} / {num_epoch:04d} | "
                      f"BATCH {batch:04d} / {int(num_batch_val):04d} | LOSS {np.mean(loss_arr):.4f}")

                label_np = fn_tonumpy(label)
                input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
                output_np = fn_tonumpy(fn_class(output))
                step = int(num_batch_val) * (epoch - 1) + batch
                writer_val.add_image('label', label_np, step, dataformats='NHWC')
                writer_val.add_image('input', input_np, step, dataformats='NHWC')
                writer_val.add_image('output', output_np, step, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir, net, optim, epoch)

    writer_train.close()
    writer_val.close()

else:
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)
            loss = fn_loss(output, label)
            loss_arr.append(loss.item())

            print(f"TEST: BATCH {batch:04d} / {int(num_batch_test):04d} | LOSS {np.mean(loss_arr):.4f}")

            label_np = fn_tonumpy(label)
            input_np = fn_tonumpy(fn_denorm(input, 0.5,_
