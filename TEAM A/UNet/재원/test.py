# 라이브러리 임포트 및 설정
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms

# 모델, 데이터셋, 유틸 함수 import
from model import AttentionUNet
from dataset import PersonSegDataset, ResizeWithPadding, ToTensor, Normalize
from util import load_checkpoint, compute_iou

# 경로 및 파라미터 설정
base_dir     = '/home/work/XAI_BASE/BASE_5주차'
data_dir     = os.path.join(base_dir, 'npy_split', 'test')
ckpt_dir     = os.path.join(base_dir, 'checkpoint')
result_dir   = os.path.join(base_dir, 'result')
batch_size   = 16
lr           = 0.001

# 결과 디렉토리 생성
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# 데이터 전처리(Transform) 구성(테스트 단계: augmentation 제외)
transform = transforms.Compose([
    ResizeWithPadding((256, 256)),  # 크기 조정 및 패딩
    ToTensor(),                     # NumPy → Tensor
    Normalize(mean=0.5, std=0.5),  # 정규화
])

# DataLoader 생성
test_dataset = PersonSegDataset(data_dir=data_dir, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_samples = len(test_dataset)
num_batches = int(np.ceil(num_samples / batch_size))

# 네트워크 및 손실함수/Optimizer 설정
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = AttentionUNet().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 체크포인트 로드
model, optimizer, start_epoch = load_checkpoint(ckpt_dir, model, optimizer)
print(f"Loaded checkpoint from epoch {start_epoch}")

# 테스트 루프
model.eval()
total_iou = 0.0
all_losses = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader, 1):
        images = data['image'].to(device)
        masks  = data['mask'].to(device)

        logits = model(images)
        loss   = criterion(logits, masks)
        all_losses.append(loss.item())

        # 확률 및 이진 마스크 생성
        probs      = torch.sigmoid(logits)
        preds_bin  = (probs > 0.5).float()

        # 로그 출력
        print(f"[TEST] Batch {batch_idx:02d}/{num_batches:02d} Loss: {np.mean(all_losses):.4f}")

        # Numpy 변환
        masks_np   = masks.cpu().numpy().transpose(0, 2, 3, 1)
        inputs_np  = (images.cpu().numpy().transpose(0, 2, 3, 1) * 0.5) + 0.5  # 정규화 해제
        preds_np   = preds_bin.cpu().numpy().transpose(0, 2, 3, 1)

        # 저장 및 IoU 계산
        for i in range(preds_np.shape[0]):
            idx = (batch_idx - 1) * batch_size + i
            if idx >= num_samples:
                break

            # PNG 저장
            plt.imsave(os.path.join(result_dir, 'png', f'label_{idx:04d}.png'), masks_np[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{idx:04d}.png'), inputs_np[i].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{idx:04d}.png'), preds_np[i].squeeze(), cmap='gray')

            # NPY 저장
            np.save(os.path.join(result_dir, 'numpy', f'label_{idx:04d}.npy'), masks_np[i].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{idx:04d}.npy'), inputs_np[i].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{idx:04d}.npy'), preds_np[i].squeeze())

            # IoU 계산
            iou = compute_iou(preds_np[i].squeeze(), masks_np[i].squeeze())
            total_iou += iou

# 최종 결과 출력
avg_loss = np.mean(all_losses)
avg_iou  = total_iou / num_samples if num_samples > 0 else 0
print(f"\n==> Average Test Loss: {avg_loss:.4f}")
print(f"==> Average IoU:   {avg_iou:.4f}")
