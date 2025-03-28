import torch
from model import VGG
from dataset import get_dataloader
from train import train
from test import evaluate

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 로드
train_loader, test_loader = get_dataloader(batch_size=100)

# 모델 초기화
model = VGG(base_dim=64).to(device)

# 학습 및 평가
train(model, train_loader, device, num_epochs=100, lr=0.0002)
evaluate(model, test_loader, device)

# Test Accuracy: 89.94% ! 아싸~