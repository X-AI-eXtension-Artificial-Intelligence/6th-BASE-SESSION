import os
import numpy as np
import torch
import torch.nn as nn

# 데이터셋 클래스
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        lst_data = os.listdir(self.data_dir)
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) / 255.0
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])) / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}
        return self.transform(data) if self.transform else data

# 텐서 변환
class ToTensor(object):
    def __call__(self, data):
        label = data['label'].transpose((2, 0, 1)).astype(np.float32)
        input = data['input'].transpose((2, 0, 1)).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

# 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = (data['input'] - self.mean) / self.std
        return {'label': data['label'], 'input': input}

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.5:
            label, input = np.fliplr(label), np.fliplr(input)
        if np.random.rand() > 0.5:
            label, input = np.flipud(label), np.flipud(input)
        return {'label': label, 'input': input}
