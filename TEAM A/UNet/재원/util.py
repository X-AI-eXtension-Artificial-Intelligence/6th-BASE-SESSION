import os
import numpy as np
import torch

# 학습된 네트워크 가중치 저장
# 모델 저장 및 로딩, IoU 계산 함수 제공

def save_checkpoint(ckpt_dir, model, optimizer, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, f"model_epoch{epoch}.pth")
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, filename)


def load_checkpoint(ckpt_dir, model, optimizer):
    if not os.path.isdir(ckpt_dir):
        return model, optimizer, 0

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpts:
        return model, optimizer, 0

    ckpts.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    checkpoint = torch.load(latest_ckpt, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint.get('epoch', 0)

    return model, optimizer, start_epoch


def compute_iou(pred_mask, true_mask, threshold=0.5):
    # Numpy 변환
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()

    # 예측 이진화
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    true_bin = (true_mask > threshold).astype(np.uint8)

    # Flatten
    pred_flat = pred_bin.flatten()
    true_flat = true_bin.flatten()

    # 합,교집합 계산
    intersection = np.logical_and(pred_flat, true_flat).sum()
    union = np.logical_or(pred_flat, true_flat).sum()

    if union == 0:
        return 1.0
    return intersection / union
