import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import trange
from vggnet import VGG
import matplotlib.pyplot as plt

#Device 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU가 있다면 GPU 사용, 없으면 CPU 사용
print(f"Using device: {device}")  # 현재 사용 중인 장치 출력

#데이터 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 픽셀 값을 -1~1 범위로 정규화
])

#CIFAR-10 데이터셋 로드
batch_size = 100  # 배치 크기 설정
cifar10_train = datasets.CIFAR10(root="./data", train=True, transform=transform, target_transform=None, download=True)  # 학습 데이터셋 로드
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)  # 학습 데이터 배치 로드

# 모델 초기화 및 학습 설정
learning_rate = 0.0002  # 학습률 0.0002
num_epoch = 50  # 학습 횟수

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

# 학습된 모델 저장
torch.save(model.state_dict(), "vgg_cifar10.pth")
print("Model saved as vgg_cifar10.pth")