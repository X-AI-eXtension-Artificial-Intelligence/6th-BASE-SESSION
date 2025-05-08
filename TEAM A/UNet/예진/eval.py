# eval.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import UNet
from dataset import *
from util import *
from torchvision import transforms

lr = 1e-3
batch_size = 4
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ▼ IoU 계산 함수 (train.py 동일하게 복사)
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

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
dataset_test = Dataset(os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

net = UNet().to(device)
fn_loss = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

net, optim, st_epoch = load(ckpt_dir, net, optim)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

with torch.no_grad():
    net.eval()
    loss_arr = []
    iou_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device).squeeze(1)  # (B, H, W)
        input = data['input'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        pred = torch.argmax(output, dim=1)
        iou = compute_iou(pred, label)
        iou_arr.append(iou)

        print("TEST: BATCH %04d / %04d | LOSS %.4f | IOU %.4f" %
              (batch, len(loader_test), np.mean(loss_arr), np.mean(iou_arr)))

        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        label_np = label.cpu().numpy()
        pred_np = pred.cpu().numpy()

        for j in range(label_np.shape[0]):
            id = (batch - 1) * batch_size + j
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label_np[j], cmap='tab10')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), pred_np[j], cmap='tab10')

            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label_np[j])
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input_np[j])
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), pred_np[j])

    print("AVERAGE TEST: LOSS %.4f | IOU %.4f" % (np.mean(loss_arr), np.mean(iou_arr)))
