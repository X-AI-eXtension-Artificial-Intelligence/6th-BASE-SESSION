import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset): # Pytorch Dataset 클래스 상속받아 사용자 정의 데이터셋 구현현
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir # 데이터 저장된 경로
        self.transform = transform # transform 함수 저장

        lst_data = os.listdir(self.data_dir) # 디렉토리 내 파일 list 가져오기기
        # label로 시작하는 파일 목록 생성
        lst_label = [f for f in lst_data if f.startswith('label')]
        # input으로 시작하는 파일 목록 생성
        lst_input = [f for f in lst_data if f.startswith('input')]
        # 정렬
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input
    # 데이터셋의 총 샘플 개수 반환
    def __len__(self):
        return len(self.lst_label)
    # 인덱스에 해당하는 라벨과 입력 데이터 load
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        # 이미지 데이터 [0,1] 범위로 정규화
        label = label/255.0
        input = input/255.0
        # 2차원인 경우 차원을 추가하여 (H,W,1) 형태로 변환
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 데이터는 dictionary 형태태
        # transform함수를 통해 변환환
        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        # Numpy -> Pytorch Tensor로 변환환
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
# 입력 이미지를 평균(mean), 표준편차(std)로 정규화하는 클래스스
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data
# 입력 이미지와 라벨을 무작위로 좌우, 상하 flip
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # 50% 확률로 진행행
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

