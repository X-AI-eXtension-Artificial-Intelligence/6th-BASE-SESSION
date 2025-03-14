# 라이브러리 불러오기
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn # 신경망 구축 라이브러리
import os
from model import VGG16  # VGG16 모델 import

# train 설정 값 정의
batch_size = 512  # 배치 사이즈: 한 번에 512개 이미지 처리
learning_rate = 0.05  # 학습률
num_epoch =5  # 전체 데이터셋을 10번 반복 학습

# 장치 설정 - GPU 사용 가능하면 사용, 아니면 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 현재 사용 중인 장치 출력

# 이미지 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균 0.5, 표준편차 0.5 기준 정규화
])

# CIFAR-10 데이터셋 로드
cifar10_train = datasets.CIFAR10(root='./Data/', train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root='./Data/', train=False, transform=transform, download=True)

# 데이터 로더 설정
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# 모델, 손실 함수, 옵티마이저 정의
model = VGG16(base_dim=64, num_classes=10).to(device)
loss_func = nn.CrossEntropyLoss()  # 분류 문제에 적합한 CrossEntropyLoss 사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 옵티마이저 사용

# 학습 루프
loss_arr = []  # 손실 값 저장 리스트

for epoch in range(num_epoch):  
    total_loss = 0  # 에포크당 총 손실 초기화

    for batch_idx, (image, label) in enumerate(train_loader):  
        x = image.to(device)  # 입력 이미지 GPU/CPU로 이동
        y = label.to(device)  # 정답 레이블 GPU/CPU로 이동

        optimizer.zero_grad()  # 이전 배치의 gradient 초기화
        output = model(x)  # 순전파 수행
        loss = loss_func(output, y)  # 손실 값 계산
        loss.backward()  # 역전파 (Gradient 계산)
        optimizer.step()  # 가중치 업데이트

        total_loss += loss.item()  # 미니배치 손실 누적

        # 100번째 미니배치마다 진행도 출력
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)  # 평균 손실 계산
    loss_arr.append(avg_loss)  # 모든 에포크 손실 저장

    # 10 에포크마다 중간 출력
    if epoch % 10 == 0:
        print(f'===> Epoch {epoch} - Avg Loss: {avg_loss:.4f}')

# 모델 저장 (경로 확인 후 저장)
save_path = "./models/VGG16_100.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 저장할 폴더가 없으면 생성
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
