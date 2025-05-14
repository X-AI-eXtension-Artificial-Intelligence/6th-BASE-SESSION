import os
import numpy as np
import torch

from model import UNet
from dataset import DatasetForSeg, data_transform
from utils import load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 파라미터
batch_size = 1  # 테스트는 보통 batch=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = './data'
ckpt_dir = './checkpoint'
log_dir = './log_test'

# 디렉토리 생성
os.makedirs(log_dir, exist_ok=True)

# Dataset 및 DataLoader
transform = data_transform()
dataset_test = DatasetForSeg(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# 네트워크
net = UNet(in_channel=1, out_channel=1).to(device)
net, _, _ = load(ckpt_dir, net, None)  # optimizer는 필요 없으므로 None

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# 후처리 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# 테스트 루프
net.eval()
with torch.no_grad():
    for idx, data in enumerate(loader_test):
        label = data['label'].to(device)
        input = data['input'].to(device)
        input_canny = data['input_canny'].to(device)
        output = net(input, input_canny)

        label_np = fn_tonumpy(label)
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        output_np = fn_tonumpy(fn_class(output))

        writer.add_image('label', label_np, idx, dataformats='NHWC')
        writer.add_image('input', input_np, idx, dataformats='NHWC')
        writer.add_image('output', output_np, idx, dataformats='NHWC')

        # 콘솔 출력
        print(f"Sample {idx + 1}/{len(loader_test)} saved to TensorBoard.")

writer.close()
print("✅ Test completed and results saved to TensorBoard.")
