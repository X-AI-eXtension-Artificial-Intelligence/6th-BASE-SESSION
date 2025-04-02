import os  # OS 관련 기능을 사용하기 위한 라이브러리
import numpy as np  # 넘파이 라이브러리 (현재 코드에서 사용되지 않음)

import torch  # PyTorch 라이브러리
import torch.nn as nn  # PyTorch의 신경망 관련 모듈

## 네트워크 저장하기 함수
def save(ckpt_dir, net, optim, epoch):
    """
    학습된 네트워크(모델)와 옵티마이저의 상태를 저장하는 함수.

    ckpt_dir (str): 체크포인트 저장 디렉토리 경로
    net (torch.nn.Module): 저장할 신경망 모델
    optim (torch.optim.Optimizer): 저장할 옵티마이저
    epoch (int): 현재 에포크 번호 (파일명에 포함됨)
    """
    # 저장할 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 모델과 옵티마이저의 상태를 저장 (딕셔너리 형태로 저장)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))  # 지정한 디렉토리에 'model_epochX.pth' 형식으로 저장

## 네트워크 불러오기 함수
def load(ckpt_dir, net, optim):
    """
    저장된 체크포인트에서 모델과 옵티마이저 상태를 불러오는 함수.

    인자
    ckpt_dir (str): 체크포인트 저장 디렉토리 경로
    net (torch.nn.Module): 불러올 신경망 모델
    optim (torch.optim.Optimizer): 불러올 옵티마이저

    리턴
    net (torch.nn.Module): 불러온 모델
    optim (torch.optim.Optimizer): 불러온 옵티마이저
    epoch (int): 불러온 모델의 마지막 학습 에포크 번호 (저장된 파일명을 기반으로 추출)
    """
    # 체크포인트 디렉토리가 존재하지 않으면 초기 에포크(0) 반환
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    # 체크포인트 디렉토리 내의 파일 목록 가져오기
    ckpt_lst = os.listdir(ckpt_dir)

    # 파일 이름을 기준으로 정렬 (에포크 번호 순으로 정렬하기 위해 숫자 부분을 추출하여 정렬)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 가장 마지막(최신) 체크포인트 파일 로드
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # 저장된 가중치(파라미터) 불러오기
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    # 파일명에서 에포크 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch  # 불러온 모델, 옵티마이저, 마지막 에포크 번호 반환
