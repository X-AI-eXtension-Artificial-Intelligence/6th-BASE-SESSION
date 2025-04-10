import os
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):  # 초기화
        self.data_dir = data_dir # .npy 파일들이 들어있는 폴더 경로
        self.transform = transform # 데이터 전처리

        lst_data = os.listdir(self.data_dir) # 폴더 안의 모든 파일 이름을 리스트로 들고오고, 
        
        # 파일 이름 분류 
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort() # 정렬 
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self): # 데이터 개수 반환 
        return len(self.lst_label)

    def __getitem__(self, index): # like 인덱스 
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) # label, input 파일을 넘파이 배열로 불러옴
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0 # 정규화
        input = input/255.0

        # 2차원 -> 3차원 형식으로 바꿔줌 (채널이 있는 것처럼)
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # transform 함수에서 쓰기 쉽게 input, label을 하나의 딕셔너리로 묶음

        if self.transform: # 전처리 적용
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object): # 넘파이 배열을 파이토치 텐서로 변환
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32) # (H,W,C) -> (C,H,W) 구조로 바꿈
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # 넘파이를 텐서로 변환 

        return data

class Normalization(object): # 입력 이미지 정규화 
    def __init__(self, mean=0.5, std=0.5): # 평균과 표준편차 지정 -> 흑백 이미지 범위 -1~1로 정규화하기 적합함
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std # 당연한 소리지만, input만

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): # 데이터 뒤집기
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5: # 50% 확률로 좌우 반전 
            label = np.fliplr(label).copy()
            input = np.fliplr(input).copy()

        if np.random.rand() > 0.5: # 50% 확률로 상하 반전 
            label = np.flipud(label).copy()
            input = np.flipud(input).copy()

        data = {'label': label, 'input': input}

        return data

class RandomRotate(object):
    def __init__(self, degrees=20):
        self.degrees = degrees

    def __call__(self, data):
        angle = np.random.uniform(-self.degrees, self.degrees)

        label, input = data['label'], data['input']

        # numpy → tensor → PIL → 회전 → numpy
        input = transforms.functional.rotate(torch.from_numpy(input), angle, fill=0).numpy()
        label = transforms.functional.rotate(torch.from_numpy(label), angle, fill=0).numpy()

        return {'input': input, 'label': label}