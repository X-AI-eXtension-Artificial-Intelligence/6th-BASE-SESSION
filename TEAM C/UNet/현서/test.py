import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import UNet
from dataset import Dataset, ToTensor
from utils import load, fn_tonumpy, fn_class, fn_denorm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

data_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/data'
ckpt_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/checkpoint'
result_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/result_test'
log_dir = '/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Code/U-Net/log_test'

os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드
transform = ToTensor()
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

# 데이터 개수 확인
num_data_test = len(dataset_test)
num_batch_test = int(np.ceil(num_data_test / batch_size))

# 모델 로드
net = UNet().to(device)
net, _, _ = load(ckpt_dir=ckpt_dir, net=net, optim=None)
net.eval()

# TensorBoard 설정
writer = SummaryWriter(log_dir=log_dir)

# 테스트
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
loss_arr = []

with torch.no_grad():
    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)

        loss = loss_fn(output, label)
        loss_arr.append(loss.item())

        print("TEST: BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_arr)))

        # 결과 변환
        label = fn_tonumpy(label)[0].squeeze()
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))[0].squeeze()
        output = fn_tonumpy(fn_class(output))[0].squeeze()

        # 저장
        id = batch - 1
        plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label, cmap='gray')
        plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input, cmap='gray')
        plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output, cmap='gray')

        np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label)
        np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input)
        np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output)

        # TensorBoard 기록
        writer.add_image('input', input[np.newaxis, :, :], id, dataformats='CHW')
        writer.add_image('label', label[np.newaxis, :, :], id, dataformats='CHW')
        writer.add_image('output', output[np.newaxis, :, :], id, dataformats='CHW')

# 평균 손실 출력
print("AVERAGE TEST LOSS: %.4f" % np.mean(loss_arr))
writer.close()