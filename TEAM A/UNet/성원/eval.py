import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import MoNuSegDataset
from model import AttentionUNet
from utils import load_checkpoint, fn_tonumpy, fn_denorm, fn_class, compute_iou

import matplotlib.pyplot as plt

# ----- 설정 -----
batch_size = 1
ckpt_dir = './checkpoints'
result_dir = './results/test'
log_dir = './logs/test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 결과 저장 폴더 준비
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# TensorBoard writer
writer_test = SummaryWriter(log_dir=log_dir)

# ----- 데이터 로딩 -----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_dataset = MoNuSegDataset(
    image_dir='./datasets/testing/Tissue Images',
    annotation_dir='./datasets/testing/Annotations',
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
num_batch_test = len(test_loader)

# ----- 모델 불러오기 -----
net = AttentionUNet().to(device)
optimizer = torch.optim.Adam(net.parameters())  # 형식 맞추기 위해 선언
net, optimizer, st_epoch = load_checkpoint(ckpt_dir=ckpt_dir, net=net, optimizer=optimizer)

# ----- 평가 -----
net.eval()
loss_arr = []
iou_arr = []
fn_loss = torch.nn.BCEWithLogitsLoss().to(device)

with torch.no_grad():
    for batch, (input, label) in enumerate(test_loader, 1):
        
        input, label = input.to(device), label.to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())


        input_np = fn_tonumpy(fn_denorm(input))
        label_np = fn_tonumpy(label)
        output_np = fn_tonumpy(fn_class(output, 0.1))


        # --- IoU 계산 ---
        output_bin = fn_class(output, 0.1)              # binary mask
        output_np_bin = fn_tonumpy(output_bin)       # (B, H, W, 1)
        label_np_bin = fn_tonumpy(label)             # (B, H, W, 1)

        for j in range(input_np.shape[0]):
            idx = num_batch_test * (batch - 1) + j

            # IoU 계산
            pred = output_np_bin[j].squeeze()  # (H, W)
            true = label_np_bin[j].squeeze()   # (H, W)
            iou = compute_iou(pred, true)
            print(f"[{idx}] pred.sum(): {pred.sum()}, true.sum(): {true.sum()}, iou: {iou:.4f}")
            iou_arr.append(iou)

            # PNG 저장
            plt.imsave(os.path.join(result_dir, 'png', f'input_{idx:04d}.png'), input_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'label_{idx:04d}.png'), label_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{idx:04d}.png'), output_np[j].squeeze(), cmap='gray')

            # NPY 저장
            np.save(os.path.join(result_dir, 'numpy', f'input_{idx:04d}.npy'), input_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'label_{idx:04d}.npy'), label_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{idx:04d}.npy'), output_np[j].squeeze())

            # TensorBoard에 이미지 저장 (한 번만 기록)
            if idx < 10:
                writer_test.add_image('input', input_np[j], idx, dataformats='HWC')
                writer_test.add_image('label', label_np[j], idx, dataformats='HWC')
                writer_test.add_image('output', output_np[j], idx, dataformats='HWC')

        sigmoid_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        print(f"[{idx}] sigmoid → min: {sigmoid_map.min():.4f}, max: {sigmoid_map.max():.4f}, mean: {sigmoid_map.mean():.4f}")


# 평균 loss, IOU 기록
avg_loss = np.mean(loss_arr)
avg_iou = np.mean(iou_arr)

writer_test.add_scalar('loss', avg_loss, st_epoch)
writer_test.add_scalar('iou', avg_iou, st_epoch)
writer_test.close()

print(f"AVERAGE TEST LOSS: {avg_loss:.4f}")
print(f"AVERAGE TEST IOU : {avg_iou:.4f}")
