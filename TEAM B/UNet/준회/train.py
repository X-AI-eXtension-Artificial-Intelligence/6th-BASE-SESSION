import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from data_loader import get_loader
from model import UNet


def train(model, loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 출력 크기를 마스크 크기에 맞춤 (일치하지 않으면 오류 발생)
            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
            )

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), 'unet_custom.pth')
    print("Model saved to unet_custom.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더
    loader = get_loader(batch_size=4, num_workers=0)

    # 모델
    model = UNet().to(device)

    # 클래스 수는 10개
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 시작
    train(model, loader, criterion, optimizer, device, num_epochs=2)

if __name__ == '__main__':
    main()