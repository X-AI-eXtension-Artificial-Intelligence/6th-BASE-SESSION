# 라이브러리 임포트: 기본 시스템, 이미지 처리, 수치 계산, 모델 학습 관련 패키지들
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 학습 진행률 시각화를 위한 tqdm
from tqdm.notebook import tqdm

# 사용자 정의 데이터 로더와 모델 로드
from data_loader import get_loader
from UNetMB import UNetMB

# KMeans 모델 로드 (마스크 생성 등에 사용 가능)
import joblib
kmeans = joblib.load('../XAI/UNet/week-5/kmeans_model.pkl')

# 학습 루프 정의
def train(model, loader, optimizer, criterion, num_epochs, device):
    model.train() # 모델을 학습 모드로 설정
    total_steps = len(loader) * num_epochs # 전체 학습 반복 횟수 (progress bar용)
    
    pbar = tqdm(total=total_steps, desc="Training", unit="batch")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0

        for images, masks in loader:
            # 입력 이미지와 정답 마스크를 GPU로 전송
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # 정확도 계산: 예측 vs 정답 픽셀 일치 비율
            acc = (outputs.argmax(dim=1) == masks).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

            # Progress bar 업데이트 및 현재 배치 정보 표시
            pbar.update(1)
            pbar.set_postfix({
                "Epoch": f"{epoch+1}/{num_epochs}",
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}"
            })

        # 에폭별 평균 손실 및 정확도 출력
        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)

        # epoch별 로그만 따로 출력
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
    
    pbar.close()
    torch.save(model.state_dict(), "../XAI/UNet/week-5/UNetMB.pth")
    print("Model saved successfully.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetMB().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # 이미지 및 마스크 폴더 경로 설정
    image_dir = '../XAI/UNet/week-5/dataset/train/Image'
    mask_dir = '../XAI/UNet/week-5/dataset/train/Mask'

    # 데이터 로더 생성 (batch_size=2 설정)
    train_loader = get_loader(image_dir, mask_dir, kmeans, batch_size=2)
    num_epochs = 100
    train(model, train_loader, optimizer, criterion, num_epochs, device)

if __name__ == '__main__':
    main()