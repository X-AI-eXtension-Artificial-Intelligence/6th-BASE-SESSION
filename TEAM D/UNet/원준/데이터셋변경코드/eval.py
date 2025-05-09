import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from dataset import MRIDataset
from util import Normalization, ToTensor

# 🔧 설정
batch_size = 8
eval_epoch = 5  # << 여기서 테스트할 에폭 지정
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = f'./results_epoch{eval_epoch:04d}'  # 에폭별 결과 저장

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 📁 결과 디렉토리 생성
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# 🔁 Transform 정의
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

# 📦 Dataset & Loader
dataset_test = MRIDataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

# 🧠 모델 & 로스 함수
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# 🧾 수동 에폭 모델 로드
def load_model_at_epoch(ckpt_dir, epoch, net, optim):
    ckpt_path = os.path.join(ckpt_dir, f'model_epoch{epoch:04d}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net'])
    optim.load_state_dict(checkpoint['optim'])
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    return net, optim

net, optimizer = load_model_at_epoch(ckpt_dir, eval_epoch, net, optimizer)
net.eval()

# 🔄 평가 함수
fn_tonumpy = lambda x: x.detach().cpu().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: (x > 0.5).float()

loss_arr = []

with torch.no_grad():
    for batch_idx, data in enumerate(loader_test, 1):
        input = data['input'].to(device)
        label = data['label'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        label_np = fn_tonumpy(label)
        output_np = fn_tonumpy(fn_class(output))

        for i in range(input_np.shape[0]):
            idx = (batch_idx - 1) * batch_size + i
            np.save(os.path.join(result_dir, 'numpy', f'input_{idx:04d}.npy'), input_np[i].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'label_{idx:04d}.npy'), label_np[i].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{idx:04d}.npy'), output_np[i].squeeze())

            plt.imsave(os.path.join(result_dir, 'png', f'input_{idx:04d}.png'), input_np[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'label_{idx:04d}.png'), label_np[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{idx:04d}.png'), output_np[i].squeeze(), cmap='gray')

        print(f"TEST [{batch_idx}/{len(loader_test)}] - Loss: {np.mean(loss_arr):.4f}")

print(f"✅ AVERAGE TEST LOSS @epoch {eval_epoch}: {np.mean(loss_arr):.4f}")
