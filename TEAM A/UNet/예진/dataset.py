# dataset.py
import os
import numpy as np
import torch
import torch.nn as nn

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

        # ▼ 변경됨: label 정규화 제거 & 타입 변경 (CrossEntropyLoss 사용 위해)
        label = label.astype(np.int64)  # label: (H, W) 정수형 class index

        # ▼ input: RGB 정규화 (0~1) & (C, H, W)로 변환
        input = input / 255.0
        if input.ndim == 3:
            input = input.transpose((2, 0, 1)).astype(np.float32)

        # label은 (1, H, W)로 채널 차원 추가
        if label.ndim == 2:
            label = label[np.newaxis, ...]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

# transform 클래스
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        data = {
            'label': torch.from_numpy(label),
            'input': torch.from_numpy(input)
        }
        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        return {'label': label, 'input': input}

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']  # : .numpy() 제거

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        return {'label': label.copy(), 'input': input.copy()}

