import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기

# 모델 저장 함수
def save(ckpt_dir, net, optim, epoch):
    """
    ckpt_dir: 저장할 디렉토리 경로
    net: 학습 중인 모델 (네트워크)
    optim: 옵티마이저 (파라미터 상태 포함)
    epoch: 현재 에폭 번호
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)  # 저장 경로가 없으면 생성

    # 모델과 옵티마이저의 가중치만 저장 (전체 객체 아님)
    torch.save(
        {'net': net.state_dict(), 'optim': optim.state_dict()},
        "%s/model_epoch%d.pth" % (ckpt_dir, epoch)  
    )


## 네트워크 불러오기
# 모델 불러오기 함수
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):   # 폴더가 없다면 초기값 반환
        epoch = 0
        return net, optim, epoch

    # 폴더 내 파일 리스트 가져와서 에폭 순으로 정렬
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # 숫자만 뽑아서 정렬

    # 최신 모델 파일 불러오기
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))  

    net.load_state_dict(dict_model['net'])        # 모델 파라미터 로드
    optim.load_state_dict(dict_model['optim'])    # 옵티마이저 상태 로드

    # 파일 이름에서 에폭 번호 추출 (예: model_epoch30.pth → 30)
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
