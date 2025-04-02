import numpy as np
import torch

def calculate_IOU(groundtruth_mask, pred_mask): # 0~1
    # PyTorch 텐서인 경우 numpy 변환 전에 CPU로 이동
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    if isinstance(groundtruth_mask, torch.Tensor):
        groundtruth_mask = groundtruth_mask.cpu().detach().numpy()

    # IOU 계산
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect

    # 예외 처리 (0으로 나누는 경우 방지)
    if union == 0:
        return 1.0 if intersect == 0 else 0.0

    iou = intersect / union
    return round(iou, 3)


 # 두 집합 간의 유사도를 측정하는 지표로, 주로 분할된 영역과 실제 마스크 간의 겹치는 정도를 평가하는 데 사용
def calculate_dice_coefficient(groundtruth_mask, pred_mask):
    """
    Dice coefficient를 계산하는 함수
    Returns:
        float: Dice coefficient 값 (0 ~ 1)
    """
    # PyTorch 텐서이면 CPU로 이동 후 NumPy 변환
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    if isinstance(groundtruth_mask, torch.Tensor):
        groundtruth_mask = groundtruth_mask.cpu().detach().numpy()

    # 이진화 (0, 1)
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    groundtruth_mask = (groundtruth_mask > 0.5).astype(np.float32)

    # 교집합과 합집합 계산
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask)

    # Dice coefficient 계산 (0으로 나누는 경우 예외 처리)
    if union == 0:
        return 1.0 if intersect == 0 else 0.0

    dice_coeff = (2.0 * intersect) / union
    return round(dice_coeff, 3)