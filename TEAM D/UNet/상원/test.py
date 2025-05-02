import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AttentionUNet, DiceLoss
from dataset import Dataset, Normalization, ToTensor
from util import load, fn_tonumpy, fn_denorm, fn_class

# Dice Coefficient 계산 함수
def compute_dice_coef(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice.item()

#  argparse 설정
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--data_dir', type=str, default='./datasets')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
parser.add_argument('--result_dir', type=str, default='./results')
args = parser.parse_args()

#  device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  결과 디렉토리 생성
os.makedirs(os.path.join(args.result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(args.result_dir, 'numpy'), exist_ok=True)

#  데이터셋 및 DataLoader (이미지/마스크 형식 지원)
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

test_dataset = Dataset(
    data_dir=os.path.join(args.data_dir, 'test'),
    transform=transform,
    mode='image'  # ✅ 이미지/마스크 형식 데이터셋일 경우 꼭 필요!
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

#  네트워크, 손실함수 정의
net = AttentionUNet().to(device)
dice = DiceLoss().to(device)
bce = nn.BCEWithLogitsLoss().to(device)
fn_loss = lambda output, label: 0.5 * dice(output, label) + 0.5 * bce(output, label)

#  체크포인트 불러오기
net, _, st_epoch = load(args.ckpt_dir, net, None)

#  테스트 루프 시작
net.eval()
loss_arr = []
dice_arr = []

with torch.no_grad():
    for batch, data in enumerate(test_loader, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        # Dice score 계산
        dice_score = compute_dice_coef(output, label)
        dice_arr.append(dice_score)

        print(f"TEST: BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f} | DICE {np.mean(dice_arr):.4f}")

        # numpy 변환
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        label_np = fn_tonumpy(label)
        output_np = fn_tonumpy(fn_class(output))

        # 결과 저장 (이미지 & npy)
        for j in range(label_np.shape[0]):
            id = (batch - 1) * args.batch_size + j
            plt.imsave(os.path.join(args.result_dir, 'png', f'label_{id:04d}.png'), label_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(args.result_dir, 'png', f'input_{id:04d}.png'), input_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(args.result_dir, 'png', f'output_{id:04d}.png'), output_np[j].squeeze(), cmap='gray')

            np.save(os.path.join(args.result_dir, 'numpy', f'label_{id:04d}.npy'), label_np[j].squeeze())
            np.save(os.path.join(args.result_dir, 'numpy', f'input_{id:04d}.npy'), input_np[j].squeeze())
            np.save(os.path.join(args.result_dir, 'numpy', f'output_{id:04d}.npy'), output_np[j].squeeze())

print(f"\n AVERAGE TEST LOSS: {np.mean(loss_arr):.4f}")
print(f" AVERAGE TEST Dice Coefficient: {np.mean(dice_arr):.4f}")
