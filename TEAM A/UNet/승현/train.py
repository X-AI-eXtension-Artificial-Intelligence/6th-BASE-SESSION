# -*- coding: utf-8 -*-
# train.py
import torch
import torch.nn as nn
import os
from unet import UNet
from data import get_cifar10_dataloaders
from utils import save_model, load_model
from eval import evaluate
from visualize import show_predictions

CHECKPOINT_PATH = "./checkpoints/unet.pth"
LOG_PATH = "./checkpoints/train_log.txt"
RESUME = True  # True일 경우 이전 체크포인트에서 이어서 학습

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_cifar10_dataloaders(batch_size=32, num_workers=2)
    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 0
    epochs = 5

    # 체크포인트 불러오기
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print("[INFO] 체크포인트에서 모델 불러오는 중...")
        model = load_model(model, CHECKPOINT_PATH, device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
        start_epoch = checkpoint.get("epoch", 0)

    log_file = open(LOG_PATH, "a")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            masks = torch.mean(inputs, dim=1, keepdim=True)
            masks = (masks + 1) / 2

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        val_loss = evaluate(model, testloader, device)
        log = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        print(log)
        log_file.write(log + "\n")

        # 모델 저장 (모델 + 옵티마이저 상태 + 에폭)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }, CHECKPOINT_PATH)

    log_file.close()
    show_predictions(model, testloader, device)
