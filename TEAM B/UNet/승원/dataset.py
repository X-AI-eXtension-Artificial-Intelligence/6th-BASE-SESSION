#모듈 불러오기
import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)  # 지정된 디렉토리 내 파일 리스트를 가져옴

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()  # 라벨 파일 리스트를 알파벳 순으로 정렬
        lst_input.sort()  # 입력 파일 리스트를 알파벳 순으로 정렬

        self.lst_label = lst_label  # 라벨 파일 리스트 저장
        self.lst_input = lst_input  # 입력 파일 리스트 저장

    def __len__(self):
        return len(self.lst_label) # 데이터셋의 크기 반환 (라벨 파일의 개수)

    def __getitem__(self, index):
        # 데이터셋에서 특정 인덱스의 데이터와 라벨을 가져옴
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))  # 라벨 데이터 로드
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))  # 입력 데이터 로드

        label = label/255.0  # 라벨 데이터를 0~1 범위로 정규화
        input = input/255.0  # 입력 데이터를 0~1 범위로 정규화

        if label.ndim == 2:
            label = label[:, :, np.newaxis]  # 라벨이 2D일 경우 3D로 변환
        if input.ndim == 2:
            input = input[:, :, np.newaxis]  # 입력이 2D일 경우 3D로 변환

        data = {'input': input, 'label': label}  # 입력과 라벨을 딕셔너리로 묶음

        if self.transform:
            data = self.transform(data)  # 데이터 변환 함수가 있으면 적용

        return data  # 변환된 데이터를 반환


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data): 
        label, input = data['label'], data['input'] # 데이터의 label과 input을 받아서 torch tensor로 변환하는 클래스

        label = label.transpose((2, 0, 1)).astype(np.float32)  # label 차원 순서 변경 후 float32로 변환
        input = input.transpose((2, 0, 1)).astype(np.float32)  # input 차원 순서 변경 후 float32로 변환

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}  # numpy 배열을 torch tensor로 변환

        return data  # 변환된 데이터를 반환

class Normalization(object): #평균, 표준편차 0.5 설정정
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data): # 데이터의 label과 input에 대해 정규화를 수행하는 클래스
        label, input = data['label'], data['input'] 

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): # 데이터에 대해 랜덤으로 좌우, 상하 반전을 수행하는 클래스
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