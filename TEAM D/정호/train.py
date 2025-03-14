import torch
import torch.nn as nn
import torchvision.transforms as transforms

from VGG16 import VGG16 ########
from torch.utils.data import DataLoader
from torchvision import datasets


num_epoch = 100         # 모델을 훈련시킬 epoch 수
learning_rate = 0.0002  # 학습률: 모델 가중치를 업데이트할 때 조정되는 크기
batch_size = 32         # 배치 크기: 모델을 한 번에 학습시키는 데이터의 수


transform = transforms.Compose([                           # Compose: 여러 전처리 단계를 결합하는 객체
    transforms.ToTensor(),                                 # 이미지를 PyTorch 텐서로 변환(0~1사이의 값으로 scaling)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 이미지 픽셀 값 범위 정규화(평균과 표준편차 모두 0.5로 설정)
])

cifar10_train = datasets.CIFAR10(root='./Data/',train=True,transform=transform,target_transform=None,download=True)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)


# CIFAR10 데이터셋 로딩(train)
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)


# VGG16 모델 인스턴스 생성 및 device에 할당
model = VGG16(base_dim=64).to(device)

# 손실함수 설정
loss_func = nn.CrossEntropyLoss() 

# 최적화 알고리즘으로 Adam을 사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

# 학습 과정
for i in range(num_epoch):                             # num_epoch만큼 반복
    for j, [image, label] in enumerate(train_loader):  # train_loader에서 배치사이즈 만큼 데이터룰 가져옴
        x = image.to(device)                           
        y = label.to(device)                          

        optimizer.zero_grad()                          # 그라디언트 초기화((각 배치때마다 새로운 그라디언트 계산)

        # 예측 및 손실 계산
        output = model.forward(x) 
        loss = loss_func(output, y)

        # 역전파와 가중치 업데이트
        loss.backward() 
        optimizer.step() 
    
    if i % 2 == 0:                                      # epoch이 2의 배수일 때마다 손실값 출력
        print(f'epoch {i} loss: {loss.item()}')
        loss_arr.append(loss.cpu().detach().numpy())    # detach tensor를 gradient 연산에서 분리

torch.save(model.state_dict(), "./train_model")         # 학습 완료 후 모델의 가중치 저장