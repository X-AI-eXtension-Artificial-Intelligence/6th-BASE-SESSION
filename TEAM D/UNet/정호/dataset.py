import os
import numpy as np

import torch
import torch.nn as nn

# 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):  # 데이터가 저장된 디렉토리 경로, 데이터에 적용할 변환(전처리)
        self.data_dir = data_dir
        self.transform = transform

        # 데이터 파일 목록 생성
        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]  # label로 시작하는 파일만 선택하여 파일목록 생성
        lst_input = [f for f in lst_data if f.startswith('input')]  # input로 시작하는 파일만 선택하여 파일목록 생성

        # 정렬
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    # 데이터 set 크기 반환
    def __len__(self):
        return len(self.lst_label)

    # 데이터 로드 및 전처리
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~1 사이로 정규화
        label = label/255.0
        input = input/255.0

        # 차원 조정: label과 input 데이터가 2차원인 경우 새로운 차원을 추가하여 3차원으로 변환(모든 데이터가 동일한 차원을 갖도록 함)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # 최종적으로 변환된 데이터 반환
        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


# 트렌스폼 구현하기
# numpy -> tensor로 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 차원 변경: (높이, 너비, 채널) -> (채널, 높이, 너비)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# Data Augmentation -> 이미지를 무작위로 수평/수직으로 뒤집음
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
