import os
import numpy as np

import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir                  # 데이터 경로
        self.transform = transform                # 적용할 transform (ToTensor, Normalize 등)

        lst_data = os.listdir(self.data_dir)      # 디렉토리 내 모든 파일 리스트

        # 'label'로 시작하는 파일들과 'input'으로 시작하는 파일들을 따로 정리
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 알파벳/숫자 순으로 정렬
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)                # 전체 데이터 개수 반환

    def __getitem__(self, index):
        # numpy 파일 로딩
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~255 범위를 0~1 범위로 정규화
        label = label / 255.0
        input = input / 255.0

        # 2차원 이미지라면 채널 축 추가 (H x W → H x W x 1)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # transform이 있으면 적용
        if self.transform:
            data = self.transform(data)

        return data


## Numpy 배열을 PyTorch Tensor로 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # HWC -> CHW 변환 후 float32 타입으로 변경
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # numpy → torch tensor로 변환
        data = {
            'label': torch.from_numpy(label),
            'input': torch.from_numpy(input)
        }

        return data

## Tensor의 픽셀 값 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):  # 기본값은 [0,1] 범위를 [-1,1]로 정규화하는 셋팅
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        # 정규화: (값 - 평균) / 표준편차
        # label은 정답 값이라 일반적으로 정규화하지 않음
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}
        return data

## 이미지 좌우/상하 랜덤 플립 (Data Augmentation)
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

