# 📁 Step 5: eval.py 🤗
# 학습된 모델을 불러와 테스트 데이터에 대한 예측 수행 및 결과 저장

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

from model import UNet
from dataset import *
from util import *

# 하이퍼파라미터 설정
lr = 1e-3
batch_size = 4
num_epoch = 100

# 경로 설정 (Colab 기반 경로 예시)
data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

# 결과 저장 폴더 없으면 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# GPU 사용 여부 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 네트워크, 손실함수, 옵티마이저 정의
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Transform 정의 (테스트에는 augmentation 제외)
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

# 테스트 데이터셋 로딩
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

# 부수 함수들 정의
to_numpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
denorm = lambda x, mean, std: (x * std) + mean
binarize = lambda x: 1.0 * (x > 0.5)

# 체크포인트에서 모델 불러오기
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# 테스트 시작
with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)

        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, np.ceil(len(dataset_test)/batch_size), np.mean(loss_arr)))

        label = to_numpy(label)
        input = to_numpy(denorm(input, mean=0.5, std=0.5))
        output = to_numpy(binarize(output))

        for j in range(label.shape[0]):
            id = int(np.ceil(len(dataset_test)/batch_size)) * (batch - 1) + j

            # PNG 저장
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')

            # NumPy 저장
            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output[j].squeeze())

# 전체 평균 손실 출력
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, np.ceil(len(dataset_test)/batch_size), np.mean(loss_arr)))