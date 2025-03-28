#모듈 불러오기
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

#Device 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU가 있다면 GPU 사용, 없으면 CPU 사용
print(f"Using device: {device}")  # 현재 사용 중인 장치 출력

#데이터 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#CIFAR-10 데이터셋 로드
batch_size = 100  # 배치 크기 설정
cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, target_transform=None, download=True)  # 학습 데이터셋 로드
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, target_transform=None, download=True)  # 테스트 데이터셋 로드

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)  # 학습 데이터 배치 로드
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)  # 테스트 데이터 배치 로드

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR-10 클래스 정의

#VGGNet 블록 정의
def conv_2_block(in_dim, out_dim): # 2개의 컨볼루션 레이어를 쌓은 블록
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 레이어
        nn.ReLU(),  # 활성화 함수:ReLU 사용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 레이어
        nn.ReLU(),  # 활성화 함수
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링 레이어
    )

def conv_3_block(in_dim, out_dim):  # 3개의 컨볼루션 레이어를 쌓은 블록 (VGG 구조)
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 세 번째 3x3 컨볼루션 레이어
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링 (특징 맵 크기 절반으로 축소)
    )

# VGGNet 모델 정의
class DeepVGG1x1(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(DeepVGG1x1, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block_with_1x1(2*base_dim, 4*base_dim),
            conv_3_block_with_1x1(4*base_dim, 8*base_dim),
            conv_3_block_with_1x1(8*base_dim, 16*base_dim),
            conv_3_block_with_1x1(16*base_dim, 16*base_dim)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16*base_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# 모델 초기화 및 학습 설정
learning_rate = 0.0001  # 학습률 0.0002
num_epoch = 20  # 학습 횟수

model = VGG(base_dim=64).to(device)  # 모델 초기화 후 GPU 또는 CPU로 이동
loss_func = nn.CrossEntropyLoss()  # 손실 함수 : CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 사용

# 학습 진행
loss_arr = []  # 손실값 저장 리스트
print("Training Started...")
for epoch in trange(num_epoch):  # 학습 진행 상황 표시
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # 데이터 GPU로 이동

        optimizer.zero_grad()  # 그래디언트 초기화
        outputs = model(images)  # 모델 예측
        loss = loss_func(outputs, labels)  # 손실 계산
        loss.backward()  # 역전파 수행
        optimizer.step()  # 최적화 수행

    if epoch % 2 == 0:
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}")  # 학습 진행 출력
        loss_arr.append(loss.item())

# 그래프 출
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 모델 평가
correct = 0  # 맞은 개수를 저장할 변수 초기화
total = 0  # 전체 개수를 저장할 변수 초기화
model.eval()  # 모델을 평가 모드로 전환 (드롭아웃 및 배치 정규화 비활성화)

with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    for images, labels in test_loader:  # 테스트 데이터 로더에서 배치 단위로 이미지와 라벨을 가져옴
        images, labels = images.to(device), labels.to(device)  # 데이터를 GPU 또는 CPU로 이동
        outputs = model(images)  # 모델에 입력 데이터를 넣고 예측값 출력
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스의 인덱스를 예측값으로 설정
        total += labels.size(0)  # 전체 샘플 수 누적
        correct += (predicted == labels).sum().item()  # 맞춘 개수 누적

    accuracy = 100 * correct / total  # 정확도 계산 (정확히 예측한 샘플 수 / 전체 샘플 수 * 100)
    print(f"Accuracy on Test Data: {accuracy:.2f}%")  # 정확도 출력

#기존 78.81%에서 77.40%로 성능 감소