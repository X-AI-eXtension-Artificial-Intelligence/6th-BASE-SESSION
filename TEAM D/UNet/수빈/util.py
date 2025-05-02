# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn

# 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    # 디렉토리 확인 & 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # 모델, optimizer 상태 저장
    torch.save({'net': net.state_dict(),        # 모델 파라미터(가중치, 편향) 저장
                'optim': optim.state_dict()},   # optimizer 상태(학습률, 모멘텀) 저장
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))            

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0                   # epoch = 0 으로 초기화
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 모델/옵티마이저 상태
    # 모델/옵티마이저 상태 복원
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # 파일 이름에서 에폭 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
