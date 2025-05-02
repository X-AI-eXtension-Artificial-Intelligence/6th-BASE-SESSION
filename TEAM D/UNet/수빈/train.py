import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split  ### ← 수정됨
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import UNet
from dataset import DriveDataset, ToTensor, Normalization, RandomFlip
from util import *
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
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

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    dataset_all = DriveDataset(split='train', transform=transform)  ### ← 수정됨

    # Train/Validation 분리 (80/20)
    train_size = int(0.8 * len(dataset_all))  ### ← 수정됨
    val_size = len(dataset_all) - train_size  ### ← 수정됨
    dataset_train, dataset_val = random_split(dataset_all, [train_size, val_size])  ### ← 수정됨

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)
    num_batch_train = int(np.ceil(num_data_train / batch_size))
    num_batch_val = int(np.ceil(num_data_val / batch_size))
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset_test = DriveDataset(split='test', transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

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

            print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")

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

                print(f"VALID: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else:
    with torch.no_grad():
        net.eval()
        for batch, data in enumerate(loader_test, 1):
            input = data['input'].to(device)  ### ← 수정됨 (label 제거)
            output = net(input)

            output = fn_tonumpy(fn_class(output))
            for j in range(output.shape[0]):
                id = batch * batch_size + j
                plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')  ### ← 수정됨

            print(f"TEST: BATCH {batch:04d} / {num_batch_test:04d}")
