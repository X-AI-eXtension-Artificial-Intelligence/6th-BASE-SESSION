import torch  # PyTorch 라이브러리 
import torch.nn as nn  # torch.nn: 신경망 모듈
import torch.optim as optim  # torch.optim: 최적화 알고리즘
import torchvision  # torchvision: 이미지 데이터 처리
import torchvision.datasets as datasets  # datasets: 데이터셋 불러오기
import torchvision.transforms as transforms  # transforms: 이미지 변환
from torch.utils.data import DataLoader  # DataLoader: 데이터 로딩
import matplotlib.pyplot as plt  # matplotlib: 그래프 시각화
import numpy as np  # numpy: 숫자 계산
from tqdm import tqdm  # tqdm: 진행상황 안내



# 합성곱 블록 (Conv2D + ReLU + Conv2D + ReLU + MaxPool)
# VGGNet: 합성곱(Convolution), 활성화 함수(ReLU), 풀링(Pooling) 반복적으로 사용

def conv_2_block(in_dim, out_dim): # conv_2_block: 두 개의 합성곱 층 (합성곱 2번 수행) + ReLU 활성화 함수가 있는 블록을 생성하는 함수
    return nn.Sequential(   # 입력 채널 수(in_dim)와 출력 채널 수(out_dim)를 받아서 nn.Sequential 반환
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 필터, 패딩=1 => 출력 크기 유지
        nn.ReLU(),  # 비선형성 함수 추가 => 학습 능력 높임
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 같은 채널 수로 또 한 번 합성곱
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링 => 크기 절반으로 축소
    )

def conv_3_block(in_dim, out_dim):  # conv_3_block: 세세 개의 합성곱 층 + ReLU 활성화 함수가 있는 블록을 생성하는 함수
    return nn.Sequential(   # 입력 채널 수(in_dim)와 출력 채널 수(out_dim)를 받아서 nn.Sequential 반환
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),   # 3x3 필터, 패딩=1 => 출력 크기 유지
        nn.ReLU(),   # 비선형성 함수 추가 => 학습 능력 높임
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 같은 채널 수로 두 번째 합성곱
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 같은 채널 수로 세 번째째 합성곱
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링 => 크기 절반으로 축소
    )


# VGGNet 모델 정의

class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):  
        # base_dim: 첫 번째 합성곱 계층에서 사용되는 필터 개수 (기본 값 = 64) / 
        # num_classes: 출력 클래스 개수 (마지막 출력층에서 뉴런 10개 생성)
        super(VGG, self).__init__()

        # 특징 추출 
        """
           대부분의 컬러 이미지 데이터셋(CIFAR-10, ImageNet...)은 RGB 형식 → 3채널
           흑백 이미지 (Grayscale) → 1채널 (1개의 색상값만 사용)
           적외선 이미지 → 1채널 또는 4채널 (적외선, 가시광선, 온도 등)
           Depth 데이터 → 1채널 (깊이 정보만 포함)
           다른 스펙트럼 영상 (예: 위성 이미지) → 4채널 이상 가능
        """
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),              # 입력 3채널 (RGB) → 출력 64채널 (3채널짜리 컬러 이미지를 64개의 필터 이용해 특징 추출)
            conv_2_block(base_dim, 2 * base_dim),      # 64 → 128채널
            conv_3_block(2 * base_dim, 4 * base_dim),  # 128 → 256채널
            conv_3_block(4 * base_dim, 8 * base_dim),  # 256 → 512채널
            conv_3_block(8 * base_dim, 8 * base_dim)   # 512 → 512채널
        )

        # 완전 연결층
        # Conv Layer 통과한 특징 맵을 -> 1차원 벡터로 변환해서 (기존 3차원) -> 일반적인 신경망(Dense Layer, FC Layer)에 넣고 -> 최종 클래스 예측
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim * 1 * 1, 4096),   # 512차원 → 4096차원 (Conv Layer 통과한 512개의 채널을 4096차원 벡터로 변환)
            nn.ReLU(True),   # 활성화 함수로 ReLU 사용 => 비선형성 추가
            nn.Dropout(),    # Dropout 추가 => 과적합 방지
            nn.Linear(4096, 1000),    # 4096 → 1000차원 (VGGNet은 ImageNet(1000개 클래스)용이므로 1000개 차원으로 만듦)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),  # 1000 → 최종 클래스 개수 (CIFAR-10이므로 클래수 개수 = 10개 되어야야)
        )

    def forward(self, x): # 순전파 연산: 입력값이 레이어 지나면서 가중치 연산 수행 -> 활성화 함수 적용 -> 최종 출력층에서 예측값 생성
        x = self.feature(x)  # 특징 추출 부분 통과
        x = x.view(x.size(0), -1)  # 다차원 데이터를 1차원으로 변환 (Flatten)
        x = self.fc_layer(x)  # 완전 연결층 통과
        return x

# CNN을 통과하면 512x1x1 특징맵이 나오고 -> Flatten 하면 512차원 벡터가 되고 -> nn.Linear(512, 4096)으로 변환해서 4096차원 벡터로 확장하는 것

 

# 2. 데이터 로드


batch_size = 100  # 한 번에 학습할 이미지 개수
learning_rate = 0.0002  # 학습률 
num_epoch = 10  # 반복 횟수

# 데이터 전처리: 텐서 변환 + 정규화 
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch Tensor 형식으로 변환 (PyTorch에서 신경망 모델 학습하려면 입력 데이터는 torch.Tensor 형식)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균 0.5, 표준편차 0.5로 정규화
])

# CIFAR-10 데이터셋 준비비
cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

# 데이터 로더(DataLoader) 설정
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

# CIFAR-10 클래스 이름 설정
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# 3. 모델 설정 및 학습

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능하면 GPU 사용해라
model = VGG(base_dim=64).to(device)  # VGG 모델 생성 후 GPU/CPU에 올리기

loss_func = nn.CrossEntropyLoss()  # 다중 분류 문제이므로 CrossEntropyLoss 사용
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 최적화 알고리즘 사용 (각각의 파라미터마다 학습률을 적응적으로 조정)

loss_arr = []  # 학습 과정에서 손실 값 저장

for epoch in range(num_epoch):  # 지정 epoch 횟수만큼 반복
    total_loss = 0   # 모델이 학습한 손실 값들의 총합
    model.train()    # 학습 모드

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):  # 진행 상태 표시
        images, labels = images.to(device), labels.to(device)  # 데이터를 GPU/CPU로 전송

        optimizer.zero_grad()    # 이전 gradient 초기화
        outputs = model(images)  # 모델에 입력을 넣어 예측값 출력
        loss = loss_func(outputs, labels)  # 손실(오차) 계산
        loss.backward()   # 역전파 수행
        optimizer.step()  # 가중치 업데이트

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)  # 평균 손실 계산
    loss_arr.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")  # 학습 과정 출력



# 4. 학습 손실 시각화

plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()



# 5. 모델 평가 (테스트 정확도)

correct = 0
total = 0
model.eval()  # 평가 모드로 변경

with torch.no_grad():  # 그래디언트(기울기) 계산 비활성화 -> 메모리 절약 + 연산 속도 증가
    for images, labels in test_loader:  # 테스트 데이터 로드
        images, labels = images.to(device), labels.to(device)  # GPU or CPU로 데이터 이동
        outputs = model(images) # 모델 예측값 계산
        _, predicted = torch.max(outputs, 1) # 가장 높은 값(확률)이 있는 클래스 선택
        total += labels.size(0)  # 전체 샘플 개수 업데이트
        correct += (predicted == labels).sum().item() # 맞힌 개수 업데이트

accuracy = 100 * correct / total          # 정확도 계산
print(f"Test Accuracy: {accuracy:.2f}%")  # 최종 정확도 출력