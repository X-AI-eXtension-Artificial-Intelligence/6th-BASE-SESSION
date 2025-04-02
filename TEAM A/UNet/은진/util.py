# 📁 Step 7: util.py
# 모델 저장과 로드를 담당하는 보조 함수들

import os
import torch

# 모델과 옵티마이저 상태를 저장하는 함수
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'net': net.state_dict(),          # 모델 가중치 저장
        'optim': optim.state_dict()       # 옵티마이저 상태 저장
    }, "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 저장된 모델을 불러오는 함수
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # 가장 최신 체크포인트 찾기

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])         # 모델 가중치 불러오기
    optim.load_state_dict(dict_model['optim'])     # 옵티마이저 상태 불러오기
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])  # 에포크 번호 추출

    return net, optim, epoch
