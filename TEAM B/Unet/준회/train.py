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

from tqdm.notebook import tqdm
from data_loader import get_loader  # 수정된 데이터 로더를 사용합니다.
from model import Unet
import joblib
  
kmeans = joblib.load('kmeans_model.pkl')


def train(model, loader, optimizer, criterion, num_epochs, device):
    model.train()  # 모델을 학습 모드로 설정
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)  # 진행률 바 업데이트

        avg_loss = total_loss / len(loader)
        print(f'Average Loss: {avg_loss:.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved successfully.")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # 데이터 로더 설정
    image_dir = '../XAI/Unet/dataset/train'  # 이미지 디렉토리 경로 설정
    train_loader = get_loader(image_dir, kmeans, batch_size=4)
    num_epochs = 25
    train(model, train_loader, optimizer, criterion, num_epochs, device)

if __name__ == '__main__':
    main()
