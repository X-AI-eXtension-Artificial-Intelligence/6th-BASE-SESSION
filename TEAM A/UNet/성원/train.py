import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import MoNuSegDataset
from model import AttentionUNet  # UNet 모델이 model.py에 있다고 가정
from utils import save_checkpoint, load_checkpoint, fn_tonumpy, fn_denorm, fn_class, DiceLoss  # 유틸 함수들

import matplotlib.pyplot as plt

# ----- 하이퍼파라미터 설정 -----
batch_size = 4
num_epoch = 200
learning_rate = 1e-4
log_dir = './logs'
ckpt_dir = './checkpoints'
result_dir = './results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- 데이터 로더 준비 -----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = MoNuSegDataset(
    image_dir='./datasets/training/Tissue Images',
    annotation_dir='./datasets/training/Annotations',
    transform=transform
)


# 데이터셋을 validation set으로 일부 나누고 싶으면 여기서 따로 분리 가능
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 간단히 train 데이터 재사용 (정식 val set 있으면 교체)

num_batch_train = len(train_loader)
num_batch_val = len(val_loader)

# ----- 모델, 손실함수, 옵티마이저 -----
net = AttentionUNet().to(device)

bce_loss = nn.BCEWithLogitsLoss().to(device)
dice_loss = DiceLoss().to(device)
def fn_loss(output, label):
    return bce_loss(output, label) + dice_loss(output, label)  # BEC + Dice 로 구성 

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ----- Tensorboard SummaryWriter -----
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# ----- 학습/검증 시작 -----
st_epoch = 0
train_continue = 'on'  # 저장된 모델 이어서 학습할지 여부 ("on"이면 load). 60에폭부터 이어 학습 

if train_continue == 'on':
    net, optimizer, st_epoch = load_checkpoint(ckpt_dir=ckpt_dir, net=net, optimizer=optimizer)

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, (input, label) in enumerate(train_loader, 1):
        input, label = input.to(device), label.to(device)

        output = net(input)
        loss = fn_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_arr.append(loss.item())

        print(f"TRAIN: EPOCH {epoch:04d}/{num_epoch:04d} | BATCH {batch:04d}/{num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")

        # TensorBoard 로그 저장
        step = num_batch_train * (epoch - 1) + batch

        writer_train.add_image('input', fn_tonumpy(fn_denorm(input)), step, dataformats='NHWC')
        writer_train.add_image('label', fn_tonumpy(label), step, dataformats='NHWC')
        writer_train.add_image('output', fn_tonumpy(fn_class(output)), step, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # ----- Validation -----
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, (input, label) in enumerate(val_loader, 1):
            input, label = input.to(device), label.to(device)

            output = net(input)
            loss = fn_loss(output, label)

            loss_arr.append(loss.item())

            print(f"VALID: EPOCH {epoch:04d}/{num_epoch:04d} | BATCH {batch:04d}/{num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")

            step = num_batch_val * (epoch - 1) + batch

            writer_val.add_image('input', fn_tonumpy(fn_denorm(input)), step, dataformats='NHWC')
            writer_val.add_image('label', fn_tonumpy(label), step, dataformats='NHWC')
            writer_val.add_image('output', fn_tonumpy(fn_class(output)), step, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

    # ----- 주기적으로 모델 저장 -----
    if epoch % 20 == 0:
        save_checkpoint(ckpt_dir=ckpt_dir, net=net, optimizer=optimizer, epoch=epoch)

# 최종 종료
writer_train.close()
writer_val.close()

print("Training Finished Successfully 🚀")
