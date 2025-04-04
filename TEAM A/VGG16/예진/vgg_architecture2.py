# -*- coding: utf-8 -*-
"""vgg_architecture2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17z8JHcSQZb427D1bZlF9kUhG_R2PCnCq
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 1. VGGNet 모델 정의 (최적화)

def conv_2_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def conv_3_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 64
            conv_2_block(base_dim, 2 * base_dim),  # 128
            conv_3_block(2 * base_dim, 4 * base_dim),  # 256
            conv_3_block(4 * base_dim, 8 * base_dim),  # 512
            conv_3_block(8 * base_dim, 8 * base_dim),  # 512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim * 1 * 1, 4096),  # CIFAR-10 (32x32) 맞춰서 수정
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x


# 2. 데이터 로드 (최적화)

batch_size = 256  # 증가하여 학습 속도 향상
learning_rate = 0.001  # 조금 더 빠르게 학습
num_epoch = 5  # 10 → 5 (빠른 확인용)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 데이터 증강 추가
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=4)


# 3. 모델 설정 및 학습 (최적화)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG(base_dim=64).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # Adam → AdamW로 변경
scaler = torch.cuda.amp.GradScaler()  # Mixed Precision 사용

loss_arr = []
for epoch in range(num_epoch):
    total_loss = 0
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 반정밀도 연산으로 속도 증가
            outputs = model(images)
            loss = loss_func(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_arr.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# 4. 학습 손실 시각화

plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


# 5. 모델 평가 (테스트 정확도)

correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



"""# 경량 VGGNet로 모델 변경

합성곱 필터 수 / 파라미터 수 줄어들어 계산량 감소
"""

def conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

class LightVGG(nn.Module): # 경량 VGGNet 모델
    def __init__(self, base_dim=32, num_classes=10):
        super(LightVGG, self).__init__()
        self.feature = nn.Sequential(
            conv_block(3, base_dim),  # 32
            conv_block(base_dim, base_dim * 2),  # 64
            conv_block(base_dim * 2, base_dim * 4),  # 128
            conv_block(base_dim * 4, base_dim * 8),  # 256
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 8 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


batch_size = 512
learning_rate = 0.01
num_epoch = 5

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=4)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightVGG(base_dim=32).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)  # 학습률 조정

loss_arr = []
for epoch in range(num_epoch):
    total_loss = 0
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_arr.append(avg_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")



plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()



correct = 0
total = 0
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



"""# VGG_G: FC층에 BatchNorm 추가

BatchNorm: 입력값 정규화, 데이터 분포 변하는 현상 감소, 과적합 방지
"""

def conv_2_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def conv_3_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

class VGG_G(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(VGG_G, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, base_dim * 2),
            conv_3_block(base_dim * 2, base_dim * 4),
            conv_3_block(base_dim * 4, base_dim * 8),
            conv_3_block(base_dim * 8, base_dim * 8)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 8 * 1 * 1, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

batch_size = 100
learning_rate = 0.0002
num_epoch = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG_G().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []
for epoch in range(num_epoch):
    total_loss = 0
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_arr.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG_G Training Loss")
plt.show()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

"""# VGG_H: Dropout 비율 증가 (0.5 → 0.7)

더 많은 뉴런이 랜덤하게 제거됨 → 다양한 특징 고르게 학습
"""

def conv_2_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def conv_3_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

class VGG_H(nn.Module):
    def __init__(self, base_dim=64, num_classes=10):
        super(VGG_H, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, base_dim * 2),
            conv_3_block(base_dim * 2, base_dim * 4),
            conv_3_block(base_dim * 4, base_dim * 8),
            conv_3_block(base_dim * 8, base_dim * 8)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(base_dim * 8 * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

batch_size = 100
learning_rate = 0.0002
num_epoch = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG_H().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []
for epoch in range(num_epoch):
    total_loss = 0
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_arr.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG_H Training Loss")
plt.show()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

