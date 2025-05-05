import os
import numpy as np
import torch

## -------- Transform 관련 클래스 -------- ##
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        input = (input - self.mean) / self.std
        return {'input': input, 'label': label}


class RandomFlip(object):
    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        if np.random.rand() > 0.5:
            input = np.flip(input, axis=2)  # horizontal
            label = np.flip(label, axis=2)
        if np.random.rand() > 0.5:
            input = np.flip(input, axis=1)  # vertical
            label = np.flip(label, axis=1)
        return {'input': input.copy(), 'label': label.copy()}  # .copy() 중요


class ToTensor(object):
    def __call__(self, sample):
        input, label = sample['input'], sample['label']
        input = torch.from_numpy(input).float()
        label = torch.from_numpy(label).float()
        return {'input': input, 'label': label}


## -------- 모델 저장 및 로드 -------- ##
def save(ckpt_dir, net, optim, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'model_epoch{epoch:04d}.pth')
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, ckpt_path)


def load(ckpt_dir, net, optim):
    ckpt_list = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpt_list:
        print("❌ No checkpoint found.")
        return net, optim, 0

    ckpt_list.sort()
    ckpt_path = os.path.join(ckpt_dir, ckpt_list[-1])
    ckpt = torch.load(ckpt_path)

    net.load_state_dict(ckpt['net'])
    optim.load_state_dict(ckpt['optim'])

    epoch = int(ckpt_path.split('epoch')[1].split('.pth')[0])
    print(f"✅ Loaded checkpoint: {ckpt_path}")
    return net, optim, epoch


## -------- 이미지 저장 -------- ##
import matplotlib.pyplot as plt

def save_image(save_path, image):
    """
    image: numpy array, shape (H, W, 1) or (H, W)
    """
    image = image.squeeze()
    plt.imsave(save_path, image, cmap='gray')
