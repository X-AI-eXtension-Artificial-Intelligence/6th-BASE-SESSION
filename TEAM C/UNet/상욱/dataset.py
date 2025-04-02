import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):  # PyTorch Dataset 클래스를 상속받아 사용자 정의 데이터셋 구현
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir # 데이터가 저장된 디렉토리 경로
        self.transform = transform # 변환(transform) 함수 저장

        lst_data = os.listdir(self.data_dir) # 데이터 디렉토리 내 파일 리스트 가져오기

        lst_label = [f for f in lst_data if f.startswith('label')] # "label"로 시작하는 파일 목록 생성
        lst_input = [f for f in lst_data if f.startswith('input')] # "input"으로 시작하는 파일 목록 생성

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self): # 데이터셋의 총 샘플 개수 반환
        return len(self.lst_label)

    def __getitem__(self, index): # 인덱스에 해당하는 라벨과 입력 데이터 로드
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        # 이미지 데이터를 [0, 1] 범위로 정규화
        label = label/255.0
        input = input/255.0
        # 차원이 2D인 경우 채널 차원을 추가하여 (H, W, 1) 형태로 변환
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}
        # 변환(transform) 적용
        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object): # NumPy 배열을 PyTorch Tensor로 변환하는 클래스
    def __call__(self, data):
        label, input = data['label'], data['input']
        # NumPy 배열의 차원 변경 (H, W, C) → (C, H, W) (PyTorch의 텐서 형식에 맞춤)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        # NumPy 배열을 PyTorch Tensor로 변환하여 딕셔너리에 저장
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object): # 입력 이미지를 평균(mean)과 표준편차(std)로 정규화하는 클래스
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean # 평균값 설정
        self.std = std # 표준편차 설정

    def __call__(self, data):
        label, input = data['label'], data['input']
        # 입력 데이터 정규화 (Z-score normalization)
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object): # 입력 이미지와 라벨을 무작위로 좌우 및 상하 반전시키는 클래스
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 50% 확률로 좌우 반전 수행
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        # 50% 확률로 상하 반전 수행
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

