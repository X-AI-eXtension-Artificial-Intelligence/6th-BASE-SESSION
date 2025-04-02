## 데이터 로더와 데이터를 로드할 때 필요한 트랜스폼들 저장

import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):  ## 데이터셋 클래스 정의 -> torch.utils.data.Dataset 클래스 상속 받기
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        ## 데이터 디렉토리에 있는 모든 파일의 리스트를 얻어옴
        lst_data = os.listdir(self.data_dir)

        ## 정렬
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

        ## 0~1 사이로 픽셀값을 정규화해주기 위해 255로 나눠줌
        label = label/255.0
        input = input/255.0

        ## 뉴럴 네트워크에 들어가는 모든 인풋은 3개의 axis를 가지고 있어야 함
        ## 따라서 라벨의 디멘션이 2일 경우에는 라벨의 마지막 axis를 생성해줘야 함
        if label.ndim == 2:
            label = label[:, :, np.newaxis]  ## axis 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]  ## axis 생성

        data = {'input': input, 'label': label}

        ## 트랜스폼 펑션을 데이터 로더의 아규먼트로 넣어준다면 트랜스폼 펑션을 거친 그 데이터를 리턴해야 함
        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):  ## 넘파이에서 텐서 형태로 변환하는 클래스
    def __call__(self, data):
        label, input = data['label'], data['input']

        ## 텐서로 넘겨주기 전 ...
        ## 이미지의 넘파이 차원 = (y, x, ch)
        ## 이미지의 텐서 차원 = (ch, y, x)
        ## 디맨션(차원) 순서가 다름
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):  ## 랜덤하게 좌우상하 플립
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:  ## 50% 의 확률로 
            label = np.fliplr(label)
            input = np.fliplr(input)
            ## 항상 인풋과 라벨을 동시에 똑같이 돌려줘야 함

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data