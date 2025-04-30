import os
import torch
import numpy as np
import torch.nn as nn

def save_checkpoint(ckpt_dir, net, optimizer, epoch):
    """
    모델과 옵티마이저 상태를 저장하는 함수
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{epoch:04d}.pth')
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")

def load_checkpoint(ckpt_dir, net, optimizer):
    """
    가장 마지막 checkpoint를 불러오는 함수
    """
    ckpt_list = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    ckpt_list.sort()
    latest_ckpt = ckpt_list[-1]

    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
    checkpoint = torch.load(ckpt_path)

    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from {ckpt_path}")

    return net, optimizer, epoch

def fn_tonumpy(tensor):
    """
    Torch Tensor를 numpy array로 변환
    (Batch차원은 그대로, 채널을 마지막으로)
    """
    return tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)

def fn_denorm(tensor, mean=0.5, std=0.5):
    """
    정규화 해제 (Normalization 역변환)
    """
    return tensor * std + mean

def fn_class(tensor, threshold=0.5):
    """
    모델 output을 0 또는 1로 이진 분류
    (segmentation task용)
    """
    return (torch.sigmoid(tensor) > threshold).float()


def compute_iou(pred_mask, true_mask, eps=1e-7):
    """
    pred_mask, true_mask: (N, H, W) 또는 (H, W) 크기의 numpy 배열 (0 또는 1)
    IoU = intersection / union
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = (intersection + eps) / (union + eps)
    return iou

class DiceLoss(nn.Module):  # Dice 손실함수 
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # sigmoid 적용 후 확률값으로 변환
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice