# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn
import deeplake 

# Dataset 클래스
class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.transform = transform

        # 훈련/테스트 세트 로드
        if split == 'train':
            self.ds = deeplake.load("hub://activeloop/drive-train")
        elif split == 'test':
            self.ds = deeplake.load("hub://activeloop/drive-test")
        else:
            raise ValueError("split은 'train' 또는 'test'여야 합니다.")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
    
        image = sample['rgb_images'].numpy() / 255.0
    
        if 'manual_masks/mask' in sample:
            label = sample['manual_masks/mask'].numpy() / 255.0
        elif 'mask' in sample:
            label = sample['mask'].numpy() / 255.0
        else:
            # test set에서는 label 없이 input만 리턴
            label = np.zeros_like(image[:, :, 0:1])  # 더미 1채널
    
        # 2채널 → 1채널로 슬라이싱
        if label.ndim == 3 and label.shape[2] == 2:
            label = label[:, :, 0:1]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
    
        data = {'input': image, 'label': label}
        
        if self.transform:
            data = self.transform(data)
    
        return data




# numpy → torch tensor 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

# 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        return {'label': label, 'input': input}

# augmentation (flip)
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        return {'label': label, 'input': input}
