# Dataset이랑 Transform을 사용해서 데이터를 자동으로 불러오고 전처리하는 코드

import os
import numpy as np
import torch
import torch.nn as nn


## Dataset 구현하기
# Dataset 클래스: 폴더 안에 있는 input_001.npy, label_001.npy 같은 파일들을 짝지어서 불러오는 역할

class Dataset(torch.utils.data.Dataset):           # PyTorch의 Dataset 클래스를 상속해서 Dataset 클래스 정의
    def __init__(self, data_dir, transform=None):  # 생성자: 데이터 폴더 경로와 transform 파이프라인을 받음
        self.data_dir = data_dir                   # 데이터가 저장될 디렉토리 생성
        self.transform = transform                 # transform: 이미지 데이터 전처리
                                                   # 1. 넘파이를 텐서형태로 2. 픽셀값 정규화 3. 이미지 변환 통해 학습 다양성 증가

        lst_data = os.listdir(self.data_dir)       # 디렉토리 안에 있는 모든 파일 이름을 리스트로 가져옴

        lst_label = [f for f in lst_data if f.startswith('label')]  # 'label'로 시작하는 파일만 추출
        lst_input = [f for f in lst_data if f.startswith('input')]  # 'input'으로 시작하는 파일만 추출

        lst_label.sort()                          # 파일 이름 기준 정렬 (순서 맞추기 위해)
        lst_input.sort()

        self.lst_label = lst_label                # 이 리스트들을 클래스 내부 변수로 저장
        self.lst_input = lst_input


    def __len__(self):                      # __len__: 전체 샘플 개수를 알려주는 함수
        return len(self.lst_label)          # 보통 label 개수 = input 개수니까 label 기준으로 길이 반환


    def __getitem__(self, index):           # __getitem__: index에 해당하는 데이터 한 쌍 불러오는 함수
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))  # input_003.npy, label_003.npy 같은 걸 불러옴
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0    # 255인 픽셀값을 0~1 범위로 정규화 (딥러닝 학습 잘 되게 하려고)
        input = input/255.0

        # 현재 이미지가 흑백이라 차원이 (H, W)인데, 모델에 넣을 땐 (H, W, C) 형식이 돼아하므로 채널 차원 추가해야 함
        # np.newaxis를 이용해서 (H, W, 1)로 만들어줌
        if label.ndim == 2:                          
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 후에 transform에 넘기기 좋게 딕셔너리로 input/label을 묶어줌 

        if self.transform:
            data = self.transform(data) # 만약 transform이 정의되어 있다면 전처리를 여기서 적용

        return data  # 최종적으로 input/label이 담긴 딕셔너리를 반환



## Transform 구현하기: 데이터 전처리(정규화, 텐서 변환 등)하거나 증강(랜덤 뒤집기)
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input'] # 딕셔너리에서 input/label 꺼냄
        # 이미지 배열은 보통 (H, W, C)인데, PyTorch는 (C, H, W) 형태로 받아야 함 -> 순서 변경해줌
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # 넘파이 배열 -> PyTorch 텐서로 변환

        return data

# 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): # 평균 0.5, 표준편차 0.5로 설정
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input'] 

        input = (input - self.mean) / self.std # input만 정규화, label은 유지 (label은 학습용 정답이니까 건드리면 안 됨)

        data = {'label': label, 'input': input}

        return data

# 데이터 증강
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 50% 확률로 좌우, 상하 반전 → 데이터 다양성 증가
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data