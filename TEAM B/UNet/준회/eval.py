import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import CityscapeDataset  # 데이터 로더
from model import Unet  # 모델
import joblib
  

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 경로 설정
    model_name = 'unet_model.pth'
    
    num_classes = 10  # 클래스 수 설정
    model = Unet().to(device)
    model.load_state_dict(torch.load(model_name))
    kmeans = joblib.load('kmeans_model.pkl')

    # 데이터셋 및 데이터 로더 설정
    val_dir = '../XAI/Unet/dataset/val'  # 검증 데이터셋 경로
    test_batch_size = 8
    dataset = CityscapeDataset(val_dir, kmeans)  # 레이블 모델을 num_classes로 대체
    data_loader = DataLoader(dataset, batch_size=test_batch_size)

    X, Y = next(iter(data_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model(X)
    Y_pred = torch.argmax(Y_pred, dim=1)

    # 이미지의 변환을 역으로 적용
    inverse_transform = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])

    fig, axes = plt.subplots(test_batch_size, 3, figsize=(15, 5*test_batch_size))
    iou_scores = []

    for i in range(test_batch_size):
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()

        # IOU 점수 계산
        intersection = np.logical_and(label_class, label_class_predicted)
        union = np.logical_or(label_class, label_class_predicted)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)

        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title("Landscape")
        axes[i, 1].imshow(label_class)
        axes[i, 1].set_title("Label Class")
        axes[i, 2].imshow(label_class_predicted)
        axes[i, 2].set_title("Label Class - Predicted")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')

    plt.show()

    print(f'Average IoU Score: {sum(iou_scores) / len(iou_scores):.4f}')

if __name__ == '__main__':
    main()
