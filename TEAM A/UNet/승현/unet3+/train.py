# train.py
import torch
import torch.nn as nn
import os
from unet3plus import UNet3Plus
from data import get_cifar10_dataloaders
from utils import save_model, load_model
from eval import evaluate
from visualize import show_predictions, save_prediction_grid

CHECKPOINT_PATH = "./checkpoints/unet3plus.pth"
LOG_PATH = "./checkpoints/train_log_unet3plus.txt"
RESULT_PATH = "./checkpoints/predictions_unet3plus.png"
RESUME = True  # True일 경우 이전 체크포인트에서 이어서 학습

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_cifar10_dataloaders(batch_size=32, num_workers=2)
    model = UNet3Plus(in_channels=3, out_channels=1)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 0
    epochs = 5

    # 체크포인트 불러오기
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print("[INFO] Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get("epoch", 0)

    # 반드시 마지막에 to(device) 호출
    model = model.to(device)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "a")

    for epoch in range(start_epoch, epochs):
        model.train()
        ㅁ = 0.0

        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            masks = torch.mean(inputs, dim=1, keepdim=True)
            masks = (masks + 1) / 2
            masks = masks.to(device)  # ← 추가된 부분

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        val_loss = evaluate(model, testloader, device)
        log = "Epoch {}/{} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(epoch+1, epochs, train_loss, val_loss)
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
    save_prediction_grid(model, testloader, device, save_path=RESULT_PATH)
