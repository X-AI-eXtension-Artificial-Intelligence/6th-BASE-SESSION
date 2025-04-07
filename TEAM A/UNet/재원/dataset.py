# 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets

# Dataloader 구현
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir #data 경로
        self.transform = transform #전처리

        lst_data = os.listdir(self.data_dir) #경로 내 모든 파일이름 리스트로 불러오기

        lst_label = [f for f in lst_data if f.startswith('label')] #label로 시작하면 label 리스트에 넣기
        lst_input = [f for f in lst_data if f.startswith('input')] #input으로 시작하면 image 리스트에 넣기

        #정렬
        lst_label.sort()
        lst_input.sort()

        #객체 생성
        self.lst_label = lst_label
        self.lst_input = lst_input

    #label 리스트 길이(총 개수)
    def __len__(self):
        return len(self.lst_label)

    #index에 해당하는 이미지 프레임 가져오는 method
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) #numpy 이미지 이기 때문에 np.load로 불러오기
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화(0~256 범위를 0~1 범위로 조정)
        label = label/255.0 
        input = input/255.0

        # 이미지와 레이블의 차원 = 2일 경우(흑백 이미지로 채널이 없을 경우), 새로운 채널(축) 생성 (H,W,1)로 초기화
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
            
        if input.ndim == 2:
            input = input[:, :, np.newaxis]


        #파이토치 dataset과 호환가능하게 딕셔너리 형태로 데이터 쌍으로 저장
        data = {'input': input, 'label': label}

        # transform이 정의되어 있다면 transform을 진행한 데이터 불러오기
        if self.transform:
            data = self.transform(data)

        return data

# 텐서 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        #Numpy에서는 H,W,C 형식인데 텐서는 C,H,W라 기존 2번 인덱스가 0번, 0번이 1번이 되도록 조정
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        #from_numpy로 numpy 형태를 다시 텐서로 변환해서 딕셔너리 저장
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# 이미지 평균과 표준편차로 정규화(초기화와 적용 call method)
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# 랜덤 반전 증강 기법 구현
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # np.random.rand()는 0부터 1까지중 랜덤이므로 0.5보다 크다로 짜면, 50프로 확률로 실행된다는 것을 표현 가능
        if np.random.rand() > 0.5:
            label = np.fliplr(label) #fliplr -> 좌우반전
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label) #flipud -> 상하반전
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data