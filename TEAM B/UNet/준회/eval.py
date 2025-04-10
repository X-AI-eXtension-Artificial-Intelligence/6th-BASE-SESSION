import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import get_loader  # 데이터 로더 함수 사용
from UNetMB import UNetMB  # 모델
import os
import joblib
from sklearn.cluster import KMeans
kmeans = joblib.load('../XAI/UNet/week-4/kmeans_model.pkl')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model_name = '../XAI/UNet/week-4/UNetMB.pth'
    model = UNetMB().to(device) 
    model.load_state_dict(torch.load(model_name))

    # 데이터셋 및 데이터 로더 설정
    val_image_dir = '../XAI/UNet/week-4/dataset/val/Image'  # 검증 이미지 데이터셋 경로
    val_mask_dir = '../XAI/UNet/week-4/dataset/val/Mask'    # 검증 마스크 데이터셋 경로
    test_batch_size = 1
    data_loader = get_loader(val_image_dir, val_mask_dir, kmeans, batch_size=test_batch_size)

    X, Y = next(iter(data_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model(X)
    Y_pred = torch.argmax(Y_pred, dim=1)

    # 이미지의 변환을 역으로 적용
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])

    fig, axes = plt.subplots(test_batch_size, 3, figsize=(15, 5 * test_batch_size))
    iou_scores = []

    if test_batch_size == 1:
        axes = np.expand_dims(axes, axis=0)  # 1차원 배열을 2차원 배열처럼 처리

    for i in range(test_batch_size):
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()

        # IOU 점수 계산
        intersection = np.logical_and(label_class, label_class_predicted)
        union = np.logical_or(label_class, label_class_predicted)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)

        # 1개 배치일 경우 axes[i]가 아니라 axes 사용
        ax0 = axes[i, 0] if test_batch_size > 1 else axes[0]
        ax1 = axes[i, 1] if test_batch_size > 1 else axes[1]
        ax2 = axes[i, 2] if test_batch_size > 1 else axes[2]

        ax0.imshow(landscape)
        ax0.set_title("Landscape")
        ax1.imshow(label_class)
        ax1.set_title("Label Class")
        ax2.imshow(label_class_predicted)
        ax2.set_title("Label Class - Predicted")

        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')

    plt.show()

    print(f'Average IoU Score: {sum(iou_scores) / len(iou_scores):.4f}')

if __name__ == '__main__':
    main()
