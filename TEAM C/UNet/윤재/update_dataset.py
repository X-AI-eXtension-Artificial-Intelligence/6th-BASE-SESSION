import os
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
import random

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0  ## 입력 및 라벨 데이터 정규화 (0~255 -> 0~1)
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]  ## (height, width) -> (height, width, channel)
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) ## (height, width, channel) -> (channel, height, width)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)  ## 좌우로 Flip
            input = np.fliplr(input)  

        if np.random.rand() > 0.5:
            label = np.flipud(label)  ## 상하로 Flip
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

class RandomRotate(object):
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, data):
        label, input = data['label'], data['input']

        # float64 -> float32 변환 (Tensor가 아니라면 numpy에서 직접 변환)
        if isinstance(input, torch.Tensor):
            input = input.float()
            input = input.numpy()
        if isinstance(label, torch.Tensor):
            label = label.float()
            label = label.numpy()

        input = input.astype(np.float32)
        label = label.astype(np.float32)

        # numpy -> PIL
        input_img = transforms.ToPILImage()(input.squeeze())
        label_img = transforms.ToPILImage()(label.squeeze())

        angle = random.uniform(-self.degrees, self.degrees)

        input_img = transforms.functional.rotate(input_img, angle, interpolation=InterpolationMode.BILINEAR)
        label_img = transforms.functional.rotate(label_img, angle, interpolation=InterpolationMode.NEAREST)

        # PIL -> numpy (float32로 명시)
        input = np.array(input_img).astype(np.float32)[..., np.newaxis]
        label = np.array(label_img).astype(np.float32)[..., np.newaxis]

        return {'input': input, 'label': label}



class GaussianBlurTensor(object):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpy array일 경우 tensor로 변환
        if isinstance(input, np.ndarray):
            input = input.transpose((2, 0, 1)).astype(np.float32)
            input = torch.from_numpy(input)
        elif isinstance(input, torch.Tensor):
            if input.ndim == 3 and input.shape[0] != 1 and input.shape[0] != 3:
                input = input.permute(2, 0, 1).contiguous()
            if input.dtype != torch.float32:
                input = input.float()  # 🔹 여기 추가 (안전성 향상)

        # 예시: label이 numpy array일 때
        if isinstance(label, np.ndarray):
            label = np.transpose(label, (2, 0, 1))  # (H, W, 1) → (1, H, W)
            label = torch.from_numpy(label).float()

        elif isinstance(label, torch.Tensor):
            if label.ndim == 3 and label.shape[-1] == 1:
                label = label.permute(2, 0, 1).contiguous()  # (H, W, 1) → (1, H, W)
            if label.dtype != torch.float32:
                label = label.float()

        if random.random() < self.p:
            input = TF.gaussian_blur(input, kernel_size=self.kernel_size, sigma=self.sigma)

        data = {'label': label, 'input': input}
        return data
