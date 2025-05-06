# ── 라이브러리 임포트 및 설정 ───────────────────────────────────
import os
import gc  # 가비지 컬렉션
import numpy as np

# CUDA 메모리 최적화(분할 할당 최대 크기 및 GC 임계치 설정)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler  # AMP support

# 모델, 데이터셋, 유틸 함수 import
from model import AttentionUNet
from dataset import PersonSegDataset, ResizeWithPadding, ToTensor, Normalize, RandomFlip
from util import save_checkpoint, load_checkpoint

# ── 학습 파라미터 설정 ─────────────────────────────────────────────
lr           = 0.001   # 학습률
batch_size   = 16      # 배치 크기 (유지)
num_epochs   = 10      # 총 에포크 수

# ── 경로 설정 및 디렉토리 생성 ─────────────────────────────────────
base_dir  = '/home/work/XAI_BASE/BASE_5주차'
data_dir  = os.path.join(base_dir, 'npy_split')
ckpt_dir  = os.path.join(base_dir, 'checkpoint')
os.makedirs(ckpt_dir, exist_ok=True)

# ── 데이터 전처리(Transform) 구성 ───────────────────────────────────
transform = transforms.Compose([
    ResizeWithPadding((256, 256)),   # 크기 변경 및 패딩
    RandomFlip(),                     # 좌우/상하 랜덤 뒤집기
    ToTensor(),                       # NumPy → PyTorch Tensor
    Normalize(mean=0.5, std=0.5),    # 정규화
])

# ── DataLoader 생성 ─────────────────────────────────────────────────
train_dataset = PersonSegDataset(
    data_dir=os.path.join(data_dir, 'train'),
    transform=transform
)
train_loader  = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # 메모리 고정으로 전송 최적화
)

val_dataset = PersonSegDataset(
    data_dir=os.path.join(data_dir, 'val'),
    transform=transform
)
val_loader  = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ── 모델·손실함수·Optimizer·Scaler 설정 ─────────────────────────────
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = AttentionUNet().to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scaler    = GradScaler()

# ── 체크포인트 로드 ─────────────────────────────────────────────────
model, optimizer, start_epoch = load_checkpoint(ckpt_dir, model, optimizer)
best_loss = float('inf')

# ── 학습 루프 ───────────────────────────────────────────────────────
for epoch in range(start_epoch + 1, num_epochs + 1):
    model.train()
    train_losses = []

    for batch_idx, data in enumerate(train_loader, 1):
        images = data['image'].to(device, non_blocking=True)
        masks  = data['mask'].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        # loss 값을 미리 저장
        loss_value = loss.item()

        # Mixed Precision: backward & step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 메모리 클리어링
        del images, masks, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

        train_losses.append(loss_value)
        print(f"[TRAIN] Epoch {epoch:02d}/{num_epochs:02d} "
              f"Batch {batch_idx:03d}/{len(train_loader):03d} "
              f"Loss: {np.mean(train_losses):.4f}")

    # ── 에포크 끝난 뒤 캐시·GC 정리 ────────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()

    # ── 검증 루프 ───────────────────────────────────────────────────
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader, 1):
            images = data['image'].to(device, non_blocking=True)
            masks  = data['mask'].to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                loss   = criterion(logits, masks)

            # 메모리 정리
            del images, masks, logits
            torch.cuda.empty_cache()
            gc.collect()

            val_losses.append(loss.item())
            print(f"[VALID] Epoch {epoch:02d}/{num_epochs:02d} "
                  f"Batch {batch_idx:03d}/{len(val_loader):03d} "
                  f"Loss: {np.mean(val_losses):.4f}")

    avg_val_loss = np.mean(val_losses)
    print(f"--> Epoch {epoch:02d} Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        save_checkpoint(ckpt_dir, model, optimizer, epoch)
        print(f"*** Best model saved at Epoch {epoch:02d}, Loss: {best_loss:.4f} ***")
