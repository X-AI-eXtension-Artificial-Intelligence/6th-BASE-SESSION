#모듈 불러오기기
import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
## 신경망 모델과 옴티마이저 Epoch를 저장하는 용
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
## 저장된 신경망 모델과 옴티마이저 Epoch를 불어오는 용용
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0 #체크포인크 디렉토리가 존재하지 않으면 0으로 반환환
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir) #파일 목록 불러오기기
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch  # 불러온 모델, 옵티마이저, 마지막 epoch 반환