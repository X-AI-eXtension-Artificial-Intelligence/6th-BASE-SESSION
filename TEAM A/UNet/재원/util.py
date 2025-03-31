# 라이브러리
import os
import numpy as np

import torch
import torch.nn as nn

# 학습된 네트워크 가중치 저장
def save(ckpt_dir, net, optim, epoch):

    # 없으면 경로 만들기
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # net으로 가중치 저장, optim으로 optimizer 저장 (딕셔너리 형태)
    # 모델명에 epoch 기록 
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 loading
def load(ckpt_dir, net, optim):

    # 만약에 저장된 경로 없으면 초기 상태이므로 epoch 0으로 초기화하고 네트워크랑 optim 상태는 그냥 그대로 return
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch


    ckpt_lst = os.listdir(ckpt_dir)

    # 리스트 폴더는 있어도 저장 파일이 없으면 초기화 하도록 기존 코드에 추가
    if len(ckpt_lst) == 0 :
        epoch = 0
        return net, optim, epoch

    # 그렇지 않으면 폴더안에 모든 저장된 모델 불러와서, epoch 번호만 필터링해서 에포크 순서대로 정렬
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 인덱스 -1의 가장 최근 epoch 모델 불러오기
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # load_state_dict로 net, optim 상태 불러오기
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    
    # 파일명 내에서 문자열 split으로 epoch 번호만 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch