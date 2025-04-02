# 튜닝된 UNet 테스트 코드 (test_tuned.py)
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from dataset import *
from util import load

data_dir = './datasets'
ckpt_dir = './checkpoint_tuned'
result_dir = './results_tuned'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

dataset_test = Dataset(os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4)


class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

fn_loss = DiceLoss().to(device)


net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-4)
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)


with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print(f"TEST: BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")

        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = (batch - 1) * 4 + j  # 고유 ID 생성

            # 이미지 저장
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')

            # 넘파이 저장
            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output[j].squeeze())


total_loss = np.mean(loss_arr)
print(f"AVERAGE TEST LOSS: {total_loss:.4f}")

with open(os.path.join(result_dir, 'loss_log.txt'), 'w') as f:
    f.write(f"AVERAGE TEST LOSS: {total_loss:.6f}\n")
