# 라이브러리 추가
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from model import UNet
from dataset import DriveDataset, ToTensor, Normalization

# 설정
batch_size = 4
ckpt_dir = './checkpoint'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 로드 (테스트셋)
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
dataset_test = DriveDataset(split='test', transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

# 모델 생성 및 불러오기
net = UNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
print(f"모델 model_epoch{st_epoch}.pth 불러오기 완료")

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.02)

# 테스트 루프
with torch.no_grad():
    net.eval()
    for batch, data in enumerate(loader_test, 1):
        input = data['input'].to(device)  # label 대신 input만 사용

        output = net(input)
        output = torch.sigmoid(output) 
        print(f"Batch {batch} Output min: {output.min().item()}, max: {output.max().item()}, mean: {output.mean().item()}")


        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(output.shape[0]):
            id = (batch - 1) * batch_size + j
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input[j].squeeze())
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output[j].squeeze())

print("테스트셋 예측 및 저장 완료")
