import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # 폴더 
        self.transform = transform  # 데이터에 적용할 변환 

        lst_data = os.listdir(self.data_dir)  # 폴더 안의 파일 목록 

        lst_label = [f for f in lst_data if f.startswith('label')]  # label 파일 리스트로 
        lst_input = [f for f in lst_data if f.startswith('input')]  # input 파일 리스트로 
        lst_label.sort()  # 정렬 -> input-label 순서 맞추기 
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):  # input-label 의 개수 
        return len(self.lst_label)

    def __getitem__(self, index):  # index에 해당하는 데이터 반환 
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))  # label 불러오기 
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))  # input 불러오기 

        label = label/255.0  # 0~1 사이로 정규화 
        input = input/255.0

        if label.ndim == 2:  # 2차원 이면(흑백 이미지라면). channel 차원이 빠져있을 수 도 있기에 
            label = label[:, :, np.newaxis]  # 3차원으로 변환. H, W, 1
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:  # transform이 정의돼 있다면, 적용 
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):  # 텐서로 변환 
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)  # (H, W, C) → (C, H, W). 일반적인 이미지 형식에서 pytorch의 이미지 형식으로 
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}  # 텐서화 , 딕셔너리에 담아줌 

        return data

class Normalization(object):  # 정규화 
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std  # 평균, std로 정규화 

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):  # 상하좌우 반전 
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)  # 좌우반전 메서드 
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)  # 상하반전 메서드 
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

