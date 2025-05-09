import pandas as pd
from utils import rle_decode
from dataset import NucleiDataset
from model import UNet
import os
import torch
from torch.utils.data import DataLoader
import numpy as np


# 테스트셋 준비
test_root = './datasets/stage1_test'
solution_df = pd.read_csv('./datasets/stage1_solution.csv')

test_ids = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
test_dataset = NucleiDataset(test_root, test_ids)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).astype(np.float32) 
    target = (target > 0.5).astype(np.float32) 
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def dice_coef(pred, target, eps=1e-7):
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.eval()

dice_scores = []
iou_scores = []

for i, data in enumerate(test_loader):
    image_id = test_ids[i]
    x = data['input'].to(device)
    with torch.no_grad():
        y_hat = model(x)
    pred_mask = (y_hat[0,0].cpu().numpy() > 0.5).astype(np.uint8)
    # 정답 RLE 복원
    rle_list = solution_df[solution_df['ImageId']==image_id]['EncodedPixels']
    gt_mask = np.zeros(pred_mask.shape, dtype=np.uint8)
    for rle in rle_list:
        gt_mask |= rle_decode(rle, pred_mask.shape)

    # IoU, Dice 계산
    iou = iou_score(pred_mask, gt_mask)
    dice = dice_coef(pred_mask, gt_mask)
    iou_scores.append(iou)
    dice_scores.append(dice)

print(f"\n평균 Dice: {np.mean(dice_scores):.4f}")
print(f"평균 IoU: {np.mean(iou_scores):.4f}")