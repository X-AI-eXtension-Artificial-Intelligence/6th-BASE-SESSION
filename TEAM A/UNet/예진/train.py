# train.py
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

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_epoch", default=10, type=int)
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

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# ▼ IoU 계산 함수 추가
def compute_iou(pred, target, num_classes=10):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    dataset_train = Dataset(os.path.join(data_dir, 'train'), transform=transform)
    dataset_val = Dataset(os.path.join(data_dir, 'val'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    num_batch_train = np.ceil(len(dataset_train) / batch_size)
    num_batch_val = np.ceil(len(dataset_val) / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    dataset_test = Dataset(os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    num_batch_test = np.ceil(len(dataset_test) / batch_size)

net = UNet().to(device)

# ▼ CrossEntropyLoss로 변경
fn_loss = nn.CrossEntropyLoss().to(device)

optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

st_epoch = 0

if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []
        iou_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device).squeeze(1)  # (B, H, W)
            input = data['input'].to(device)

            output = net(input)  # (B, C, H, W)
            loss = fn_loss(output, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

            pred = torch.argmax(output, dim=1)
            iou = compute_iou(pred, label)
            iou_arr.append(iou)

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | IOU %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr), np.mean(iou_arr)))

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        writer_train.add_scalar('iou', np.mean(iou_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []
            iou_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device).squeeze(1)
                input = data['input'].to(device)
                output = net(input)
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                pred = torch.argmax(output, dim=1)
                iou = compute_iou(pred, label)
                iou_arr.append(iou)

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | IOU %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr), np.mean(iou_arr)))

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
            writer_val.add_scalar('iou', np.mean(iou_arr), epoch)

        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
