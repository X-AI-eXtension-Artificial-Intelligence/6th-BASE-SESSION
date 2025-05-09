import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_custom import CityscapesCombinedDataset
from model import UNet  # 이미 정의된 UNet 모델 import

import os

## 트레이닝 파라메터 설정하기
batch_size = 4
num_epochs = 20
learning_rate = 1e-4

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 데이터셋 및 데이터로더
train_dataset = CityscapesCombinedDataset(
    root_dir="./datasets_city/train",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실함수, 옵티마이저 정의
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary segmentation일 경우
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

# 모델 저장
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/unet_cityscapes.pth")
