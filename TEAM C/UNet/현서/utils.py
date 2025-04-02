import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch},
               os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % epoch))

def load(ckpt_dir, net, optim=None):
    ckpt_list = os.listdir(ckpt_dir)
    if not ckpt_list:
        return net, optim, 0

    ckpt_list.sort()
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    
    # ✅ optim이 있을 때만 로드
    if optim is not None:
        optim.load_state_dict(dict_model['optim'])

    epoch = dict_model['epoch']
    return net, optim, epoch




import numpy as np
import os

def generate_dummy_data(save_dir, num_samples=5, image_size=(256, 256)):
    """
    테스트용 input/label .npy 파일을 생성하는 함수.

    Parameters:
    - save_dir (str): 데이터를 저장할 디렉토리 경로
    - num_samples (int): 생성할 데이터 샘플 개수
    - image_size (tuple): (height, width) 이미지 크기
    """
    os.makedirs(save_dir, exist_ok=True)

    if len(os.listdir(save_dir)) > 0:
        print(f"⚠️ 이미 {save_dir} 경로에 파일이 존재합니다. 생성하지 않았습니다.")
        return

    for i in range(num_samples):
        input_arr = np.random.randint(0, 256, image_size, dtype=np.uint8)
        label_arr = np.random.randint(0, 2, image_size, dtype=np.uint8) * 255

        np.save(os.path.join(save_dir, f'input_{i:03d}.npy'), input_arr)
        np.save(os.path.join(save_dir, f'label_{i:03d}.npy'), label_arr)

    print(f"✅ {num_samples}개의 더미 input/label .npy 파일이 {save_dir}에 생성되었습니다.")


# utils.py 안에 추가

def fn_tonumpy(x):
    return x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

def fn_class(x):
    return 1.0 * (x > 0.5)

def fn_denorm(x, mean, std):
    return (x * std) + mean
