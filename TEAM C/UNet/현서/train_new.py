import os
import numpy as np
import torch
import torch.nn as nn

from model import UNet
from dataset import DatasetForSeg, data_transform
from utils import save, load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 훈련 파라미터
lr = 1e-3
batch_size = 4
num_epoch = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = './data'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './result'

# 디렉토리 생성
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Dataset 및 DataLoader
transform = data_transform()
dataset_train = DatasetForSeg(data_dir=os.path.join(data_dir, 'train'), transform=transform)
dataset_val = DatasetForSeg(data_dir=os.path.join(data_dir, 'val'), transform=transform)

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# 네트워크, 손실함수, optimizer
net = UNet(in_channel=1, out_channel=1).to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 부수적인 변수
num_batch_train = np.ceil(len(dataset_train) / batch_size)
num_batch_val = np.ceil(len(dataset_val) / batch_size)

# Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# load checkpoint
net, optim, st_epoch = load(ckpt_dir, net, optim)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# 학습 루프
for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)
        input_canny = data['input_canny'].to(device)
        output = net(input, input_canny)

        optim.zero_grad()
        loss = fn_loss(output, label)
        loss.backward()
        optim.step()

        loss_arr.append(loss.item())

        print(f"Train: Epoch {epoch:04d}/{num_epoch:04d} | Batch {batch:04d}/{int(num_batch_train)} | Loss {np.mean(loss_arr):.4f}")

        label_np = fn_tonumpy(label)
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        output_np = fn_tonumpy(fn_class(output))

        writer_train.add_image('label', label_np, epoch * batch, dataformats='NHWC')
        writer_train.add_image('input', input_np, epoch * batch, dataformats='NHWC')
        writer_train.add_image('output', output_np, epoch * batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # validation
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            input_canny = data['input_canny'].to(device)
            output = net(input, input_canny)

            loss = fn_loss(output, label)
            loss_arr.append(loss.item())

            label_np = fn_tonumpy(label)
            input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
            output_np = fn_tonumpy(fn_class(output))

            writer_val.add_image('label', label_np, epoch * batch, dataformats='NHWC')
            writer_val.add_image('input', input_np, epoch * batch, dataformats='NHWC')
            writer_val.add_image('output', output_np, epoch * batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
        print(f"Valid: Epoch {epoch:04d}/{num_epoch:04d} | Loss {np.mean(loss_arr):.4f}")

    if epoch % 50 == 0:
        save(ckpt_dir, net, optim, epoch)

writer_train.close()
writer_val.close()
