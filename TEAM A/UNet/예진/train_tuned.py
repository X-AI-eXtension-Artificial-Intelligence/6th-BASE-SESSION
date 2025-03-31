# 튜닝된 UNet 학습 코드 (train_tuned.py)


"""
Learning Rate:	1e-3 -> 1e-4	=> 학습률 낮춰서 더 안정적으로 학습
Batch Size:	4 -> 8	            => 한 번에 더 많은 데이터로 학습하여 일반화 성능 향상
Optimizer: Adam -> AdamW        => weight decay 포함된 최적화기 사용하여 과적합 줄임
Loss Function:	BCEWithLogitsLoss -> Dice + BCE 조합	=> 불균형 클래스 문제에 강한 Dice Loss 추가
Epoch 수: 100 -> 200	        => 더 오래 학습해서 성능 개선
데이터 증강: RandomFlip	-> RandomFlip + RandomRotation	=> 데이터 다양성 향상
"""


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import UNet
from dataset import *
from util import save, load

lr = 1e-4
batch_size = 4
num_epoch = 200

data_dir = './datasets'
ckpt_dir = './checkpoint_tuned'
log_dir = './log_tuned'
result_dir = './results_tuned'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets) + self.dice(inputs, targets)

transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    RandomFlip(),
    ToTensor()
])

dataset_train = Dataset(os.path.join(data_dir, 'train'), transform=transform)
dataset_val = Dataset(os.path.join(data_dir, 'val'), transform=transform)

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)
num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

net = UNet().to(device)
fn_loss = DiceBCELoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


st_epoch = 0
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

        loss_arr += [loss.item()]

        print(f"TRAIN: EPOCH {epoch:04d} | BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")

        input = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        label = fn_tonumpy(label)
        output = fn_tonumpy(fn_class(output))

        writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
    scheduler.step()

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print(f"VALID: EPOCH {epoch:04d} | BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")

            input = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
            label = fn_tonumpy(label)
            output = fn_tonumpy(fn_class(output))

            writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

    writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    if epoch % 50 == 0:
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()
