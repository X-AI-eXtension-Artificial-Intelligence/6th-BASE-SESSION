# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn

# 데이터 로더 구현
# 데이터 분배 더 효율적
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # data_dir 내부에 있는 파일들 가지고 오기

        # 파일 내 label 및 input 파일 각각 필터링
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 리스트 정렬 
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 데이터 픽셀 정규화(Normalization)
        # 0~255 사이 정수로 표현되어 있는 이미지 데이터 픽셀 값을 0~1 사이 실수 값으로 변환
        # 스케일링을 통해 더 빠르고 효율적인 연산 가능
        label = label/255.0
        input = input/255.0

        # 데이터 차원 확장(pytorch 특성상)
        # np.newaxis를 통해 새로운 차원 추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

# 기존 넘파이 배열에서 PyTorch 텐서로 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # numpy 차원 = (Y, X, CH)
        # tensor 차원 = (CH, Y, X)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# Normalization을 통해 안정적인 학습 구현
# 평균 & 표준편차 기준으로 스케일링
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# Data Augmentation을 통해 학습 데이터 다양화
# 여기서는 input 데이터와 label 데이터를 flip함
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:   # 랜덤 값이 0.5 초과면 적용
            label = np.fliplr(label) # 좌우 반전(Flip Left-Right)
            input = np.fliplr(input) # 좌우 반전(Flip Left-Right)

        if np.random.rand() > 0.5:   # 랜덤 값이 0.5 초과면 적용
            label = np.flipud(label) # 상하 반전(Flip Up-Down)
            input = np.flipud(input) # 상하 반전(Flip Up-Down)

        data = {'label': label, 'input': input}

        return data