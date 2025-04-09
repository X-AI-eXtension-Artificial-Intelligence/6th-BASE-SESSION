import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset): 

     # torch.utils.data.Dataset 이라는 파이토치 base class를 상속받아 
     # 그 method인 __len__(), __getitem__()을 오버라이딩 해줘서 
     # 사용자 정의 Dataset class를 선언한다
     
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)
    	
        # 문자열 검사해서 'label'이 있으면 True 
        # 문자열 검사해서 'input'이 있으면 True
        lst_label = [f for f in lst_data if f.startswith('label')] 
        lst_input = [f for f in lst_data if f.startswith('input')] 
        
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)
	
    # 여기가 데이터 load하는 파트
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

# normalize, 이미지는 0~255 값을 가지고 있어 이를 0~1사이로 scaling
        label = label/255.0
        inputs = inputs/255.0
        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)
        
# 인풋 데이터 차원이 2이면, 채널 축을 추가. 
# 파이토치 인풋은 (batch, 채널, 행, 열)

        if label.ndim == 2:  
            label = label[:,:,np.newaxis]
        if inputs.ndim == 2:  
            inputs = inputs[:,:,np.newaxis] 

        data = {'input':inputs, 'label':label}

        if self.transform:				
            data = self.transform(data)
# transform에 할당된 class 들이 호출되면서 __call__ 함수 실행

        return data

# Transform

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
		
        # numpy와 tensor의 배열 차원 순서가 다르다. 
        # numpy : (행, 열, 채널)
        # tensor : (채널, 행, 열)
        # 따라서 위 순서에 맞춰 transpose
        
        label = label.transpose((2, 0, 1)).astype(np.float32) 
        input = input.transpose((2, 0, 1)).astype(np.float32)
		
        # 이후 np를 tensor로 바꾸는 코드는 다음과 같이 간단하다.
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    

# Data Augmentation -0408 

class RandomFlip:
    def __call__(self, data):
        input, label = data['input'], data['label']
        if np.random.rand() > 0.5:
            input = np.flip(input, axis=1)
            label = np.flip(label, axis=1)
        if np.random.rand() > 0.5:
            input = np.flip(input, axis=0)
            label = np.flip(label, axis=0)
        return {'input': input.copy(), 'label': label.copy()}

class RandomRotate:
    def __call__(self, data):
        k = np.random.choice([0, 1, 2, 3])
        input = np.rot90(data['input'], k, axes=(0, 1))
        label = np.rot90(data['label'], k, axes=(0, 1))
        return {'input': input.copy(), 'label': label.copy()}

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
