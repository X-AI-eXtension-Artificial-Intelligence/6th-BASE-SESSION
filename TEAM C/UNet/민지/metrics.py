import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import label

def calculate_IOU(groundtruth_mask, pred_mask): # 0~1
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


def calculate_pixel_error(gt, pred):
    """
    전체 픽셀 중 잘못 분류된 픽셀 비율을 계산
    """
    # 이진화 (0.5 기준)
    gt_bin = (gt > 0.5).astype(np.int32)
    pred_bin = (pred > 0.5).astype(np.int32)
    
    error = np.mean(gt_bin != pred_bin)
    return round(error, 3)

def calculate_rand_error(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    
    # 두 분할 결과의 유사도를 측정 (값이 1이면 완벽 유사)
    rand_index = adjusted_rand_score(gt, pred)
    return round(1 - rand_index, 3)

def calculate_warping_error(gt, pred):
    """
    연결된 컴포넌트(객체)의 개수 차이를 기반으로 warping error를 계산
    (실제 구현은 split/merge 상황을 더 정밀하게 평가해야 함)
    """
    # 이진화 (0.5 기준)
    gt_bin = (gt > 0.5).astype(np.int32)
    pred_bin = (pred > 0.5).astype(np.int32)
    
    _, num_gt = label(gt_bin)
    _, num_pred = label(pred_bin)
    
    error = abs(num_gt - num_pred) / (num_gt + num_pred + 1e-6)
    return round(error, 3)


def calculate_errors(groundtruth_mask, pred_mask): # 
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    if isinstance(groundtruth_mask, torch.Tensor):
        groundtruth_mask = groundtruth_mask.cpu().detach().numpy()
    
    pixel_error = calculate_pixel_error(groundtruth_mask, pred_mask)
    rand_error = calculate_rand_error(groundtruth_mask, pred_mask)
    warping_error = calculate_warping_error(groundtruth_mask, pred_mask)

    return pixel_error, rand_error, warping_error
    
if __name__ == '__main__':
    pixel_error, rand_error, warping_error = calculate_errors(np.array([1,0,0,0,0,0,0,4,5,1]), np.array([1,0,0,0,0,0,0,4,5,1]))
    print(f'모두 0인가? : pixel_error: {pixel_error} ### rand_error:{rand_error} ### warping_error:{warping_error}')
    pixel_error, rand_error, warping_error = calculate_errors(np.array([3,0,0,5,0,0,0,9,5,1]), np.array([1,0,0,0,0,0,0,4,5,1]))
    print(f'모두 0이 아니여야 함 : pixel_error: {pixel_error} ### rand_error:{rand_error} ### warping_error:{warping_error}')