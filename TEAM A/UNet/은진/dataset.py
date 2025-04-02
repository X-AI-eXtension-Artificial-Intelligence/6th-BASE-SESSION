# 📁 Step 2: dataset.py 😍
# 커스텀 PyTorch Dataset 클래스 및 Transform 정의

import os
import numpy as np
import torch

# 커스텀 데이터셋 클래스 정의
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)  # 폴더 내 모든 파일 목록

        # 파일 이름에서 label과 input만 분리
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

        # 0~255 → 0~1로 정규화
        label = label / 255.0
        input = input / 255.0

        # 채널 차원 추가 (2D → 3D)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # Transform이 정의되어 있으면 적용
        if self.transform:
            data = self.transform(data)

        return data


# Transform: numpy → torch tensor
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # (H, W, C) → (C, H, W), float32로 변환
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


# Transform: 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data


# Transform: 좌우/상하 랜덤 플립
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data