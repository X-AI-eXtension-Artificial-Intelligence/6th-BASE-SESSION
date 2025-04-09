import argparse 

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 학습률 조정 : 1e-3 -> 1e-4
## Adam 계열에서는 1e-4가 더 일반적이고 성능이 좋을 때가 많음
parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
# Batch size 조정 : 4 -> 8 하려고 했는데 실패
## 평균적인 gradient가 더 잘 잡혀서 더 안정적인 학습을 가능하게 함
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
# epoch 수 조정 : 100 -> 200
## 시간이 더 걸리지만, 모델이 더 충분히 학습 가능하게 하며, 작은 배치 + 낮은 학습률 조합에서는 꼭 해야 함
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")

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

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

if mode == 'train': 
    # 데이터 증강 추가 
    ## transforms.RandomRotation(degrees=20) : 입력 이미지를 -20도 ~ +20도 사이 랜덤하게 회전시킴
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), RandomRotate(degrees=20), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4) 

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else: 
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]) 

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4) 

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

net = UNet().to(device)

fn_loss = nn.BCEWithLogitsLoss().to(device) 

# optimizer 변경 : Adam -> AdamW
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Scheduler 추가
## 초반에는 큰 학습률로 빠르게 수렴하고, 후반엔 작은 학습률로 미세하게 튜닝
## 초반에는 전반적인 패턴 빠르게 배우고, 중반 이후에는 세부 디테일 조심스럽게 조정할 수 있음
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) 
fn_denorm = lambda x, mean, std: (x * std) + mean 
fn_class = lambda x: 1.0 * (x > 0.5) 

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

st_epoch = 0 

if mode == 'train':
    if train_continue == "on": 
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) 

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

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch) 

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

        # scheduler 한 step 이동
        scheduler.step()

        if epoch % 50 == 0: 
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

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
