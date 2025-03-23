# train.py
# 이 파일은 모델 학습에 필요한 데이터셋 로드, 전처리, 모델 생성, 학습 루프 수행 및 모델 저장을 담당한다.

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# model.py에 정의된 VGG 모델 클래스를 불러온다.
from model import VGG

# -----------------------------------------------------------------------------
# 1. 데이터 전처리 및 DataLoader 설정
# -----------------------------------------------------------------------------
# 이미지 데이터를 텐서로 변환하고 정규화하기 위한 transform을 정의.
# 정규화(Normalize) 과정은은 각 채널의 픽셀값을 평균과 표준편차로 조정하여 학습의 안정성을 높인다.
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 파이토치 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5),  # RGB 각 채널의 평균값
                         (0.5, 0.5, 0.5))  # RGB 각 채널의 표준편차
])

# CIFAR10 학습 데이터셋을 다운로드하고 transform을 적용한다.
train_dataset = datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
# DataLoader를 통해 데이터셋을 배치 단위(여기서는 100개씩)로 불러오며, 데이터 순서를 섞어서(shuffle=True) 학습에 사용한다.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# -----------------------------------------------------------------------------
# 2. 디바이스 설정 및 모델 인스턴스 생성
# -----------------------------------------------------------------------------
# GPU가 사용 가능한 경우 GPU를 사용하고, 그렇지 않으면 CPU를 사용하도록 설정한다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# VGG 모델을 생성한다. base_dim은 64, CIFAR10의 경우 분류할 클래스가 10개.
model = VGG(base_dim=64, num_classes=10).to(device)

print("사용하는 디바이스:", device)

# -----------------------------------------------------------------------------
# 3. 손실함수 및 최적화 함수 설정
# -----------------------------------------------------------------------------
# 분류 문제에 사용되는 CrossEntropyLoss를 손실 함수로 사용한다.
loss_func = nn.CrossEntropyLoss()
# Adam 옵티마이저를 사용하여 모델의 파라미터를 업데이트한다. 학습률은 0.0002로 설정.
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# -----------------------------------------------------------------------------
# 4. 학습 과정 및 결과 시각화를 위한 설정
# -----------------------------------------------------------------------------
# 총 에폭(epoch) 수를 100으로 설정.
num_epoch = 100
# 에폭마다의 손실(loss) 값을 저장할 리스트.
loss_arr = []

# -----------------------------------------------------------------------------
# 5. 학습 시작
# -----------------------------------------------------------------------------
# trange를 사용하여 진행바(progress bar)를 표시하면서 에폭을 반복한다.
for epoch in trange(num_epoch, desc="학습 진행"):
    # 모델 학습습
    model.train()
    # DataLoader를 통해 배치 단위로 데이터를 가져온다.
    for image, label in train_loader:
        # 배치 내 이미지와 라벨을 지정된 디바이스(GPU 또는 CPU)로 이동.
        inputs = image.to(device)
        targets = label.to(device)

        # 옵티마이저의 기울기를 0으로 초기화하여 이전 배치의 기울기가 누적되지 않도록 한다.
        optimizer.zero_grad()
        # 모델의 forward 메소드를 호출하여 예측 결과를 계산한다.
        outputs = model(inputs)
        # 예측 결과와 실제 라벨을 비교하여 손실 값을 계산한다.
        loss = loss_func(outputs, targets)
        # 역전파를 수행하여 기울기를 계산한다.
        loss.backward()
        # 계산된 기울기를 바탕으로 가중치를 업데이트한다.
        optimizer.step()

    # 10 에폭마다 현재 에폭의 손실 값을 출력하고, 리스트에 저장한다.
    if (epoch + 1) % 10 == 0:
        print("에폭 [{}/{}], 손실값: {:.4f}".format(epoch+1, num_epoch, loss.item()))
        loss_arr.append(loss.item())

# -----------------------------------------------------------------------------
# 6. 학습 과정의 손실값 변화를 그래프로 시각화
# -----------------------------------------------------------------------------
plt.figure()  # 새로운 그래프 창을 생성한다.
# x축: 에폭 수 (저장된 손실 값의 인덱스에 10을 곱함), y축: 손실 값
plt.plot(np.arange(len(loss_arr)) * 10, loss_arr, marker='o')
plt.title("학습 손실값 변화")
plt.xlabel("에폭")
plt.ylabel("손실값")
plt.grid(True)  # 그래프에 격자선을 추가.
plt.show()  # 그래프를 화면에 출력.

# -----------------------------------------------------------------------------
# 7. 학습이 완료된 모델 저장
# -----------------------------------------------------------------------------
# torch.save 함수를 사용하여 모델의 파라미터(state_dict)를 'vgg_model.pth' 파일로 저장.
torch.save(model.state_dict(), 'vgg_model.pth')
