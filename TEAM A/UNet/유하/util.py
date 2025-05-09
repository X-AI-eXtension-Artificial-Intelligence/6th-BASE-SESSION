import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir): # 저장할 폴더가 없으면 생성하는 코드 
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch)) # 모델, optimize state 저장 

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir): # 체크포인트 폴더가 없으면, 즉 저장된 게 없으면 새로 시작
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 마지막 체크포인트 불러오기

    net.load_state_dict(dict_model['net']) # 모델, optimizer 상태 로드
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])# 저장된 epoch 번호를 숫자로 뽑아서 epoch 변수에 저장 (학습을 이어서 하기 위함)

    return net, optim, epoch

import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
