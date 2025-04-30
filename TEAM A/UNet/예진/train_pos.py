## 라이브러리 추가하기
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from dataset import *
from util_evaluation_change import *
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
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
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
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
fn_loss = nn.CrossEntropyLoss(ignore_index=255).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True)

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

st_epoch = 0
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        torch.autograd.set_detect_anomaly(True)
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device).squeeze(1).long()
            input = data['input'].to(device)

            label[label > 2] = 255  # 무시할 라벨 지정

            output = net(input)
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label_vis = fn_tonumpy(label.unsqueeze(1))
            input_vis = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
            output_vis = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label_vis, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input_vis, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output_vis, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device).squeeze(1).long()
                input = data['input'].to(device)
                label[label > 2] = 255  # 무시할 라벨 지정

                output = net(input)
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                label_vis = fn_tonumpy(label.unsqueeze(1))
                input_vis = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output_vis = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label_vis, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input_vis, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output_vis, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

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
            label = data['label'].to(device).squeeze(1).long()
            input = data['input'].to(device)
            label[label > 2] = 255

            output = net(input)
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            label_vis = fn_tonumpy(label.unsqueeze(1))
            input_vis = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output_vis = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = int((batch - 1) * batch_size + j)
                plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label_vis[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input_vis[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output_vis[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label_vis[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input_vis[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output_vis[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
