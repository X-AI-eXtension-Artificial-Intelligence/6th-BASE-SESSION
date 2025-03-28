import os
import numpy as np

import torch
import torch.nn as nn

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

        label = label/255.0
        input = input/255.0

#2차원 구조를 3차원으로 늘려주자
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object): # numpy에서 torch tensor로 변환
    def __call__(self, data):
        label, input = data['label'], data['input']

## 채널 차원을 맨 앞으로 이동 (HWC -> CHW)하고 float32 타입으로 변환
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
'파이토치에서 제공하는 Normalization 사용하려면'
'ToTensor 뒤에 위치해야 함.Normalize()가 torch tensor 형식을 입력으로 받기 때문'

## 텐서의 픽셀 값 정규화하자
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

    # 정규화: (값 - 평균) / 표준편차
    # input은 tensor 형식이어야 하므로 반드시 ToTensor 이후에 사용해야 함
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

## 이미지를 랜덤하게 좌우 혹은 상하로 뒤집자
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
    # 좌우 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
    # 상하 반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

