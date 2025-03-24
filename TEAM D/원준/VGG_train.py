import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG_model import VGG
import torch
import torch.nn as nn
from tqdm import trange


learning_rate = 0.1

# CUDA 사용 여부 확인 후 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 모델 인스턴스화 및 디바이스로 이동
model = VGG(base_dim=64).to(device) # 첫 번째 컨볼루션 레이어가 출력하는 채널 수가 64

# 손실 함수 및 최적화 기법 정의
loss_func = nn.CrossEntropyLoss()  # 다중 분류 문제에 적합한 손실 함수
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)  # Adam Optimizer 사용



# 데이터 변환 정의 (정규화 포함)
transform = transforms.Compose(
    [transforms.ToTensor(),  # 신경망에서 처리하기에 적합한 [0, 1] 범위로 변환
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 0을 중심으로 분포하도록 만들고, 학습이 더 빠르고 안정적

# CIFAR-10 학습 데이터셋 로드  
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, download=True)

# CIFAR-10 테스트 데이터셋 로드
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, download=True)

# CIFAR-10 클래스 목록
target_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# 데이터 변환 재정의 (단일 채널 평균 정규화)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 단일 채널(흑백 이미지일 경우)
])

# CIFAR-10 데이터 로드 
batch_size = 4  # 배치 크기 설정
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
# num_workers 데이터를 병렬로 로드할 때 사용되는 프로세스의 수

# 이미지 시각화를 위한 함수 정의
def imshow(img):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # PyTorch 텐서 (채널, 높이, 너비) --> matplotlib (높이, 너비, 채널)
    plt.show()

# 데이터 로더에서 무작위 배치 가져오기
dataiter = iter(train_loader)
images, labels = next(dataiter) # train_loader에서 다음 배치를 가져옴옴

# 배치에서 이미지 출력
imshow(torchvision.utils.make_grid(images))

# 해당 이미지의 클래스 출력
print(' '.join('%5s' % target_classes[labels[j]] for j in range(batch_size)))

# 학습용 데이터: 50,000개의 이미지
# 테스트용 데이터: 10,000개의 이미지    총 60000개 한 레이블 당 6000개 


batch_size = 128
learning_rate = 0.001
num_epoch = 100

loss_arr = []
for i in trange(num_epoch):  # num_epoch 만큼 반복 
    for j,[image,label] in enumerate(train_loader): 
        x = image.to(device)
        y_= label.to(device)

        optimizer.zero_grad()  # 기울기 0으로 초기화 
        output = model.forward(x)  # 입력 이미지를 모델에 전달하여 예측 값을 계산
        loss = loss_func(output,y_)
        loss.backward()  # 기울기 계산 
        optimizer.step() # 가중치 업데이트 

    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())  # 손실 값을 기록하여 후속 분석에 활용하기 위함
