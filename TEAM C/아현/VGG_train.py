import torch
import torch.nn as nn
import torchvision.transforms as transforms

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

## GPU 사용 가능 여부에 따라 모델 훈련을 위한 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 학습 변수 설정
num_epoch = 4  ## 모델을 훈련시킬 총 에폭 수 -> 전체 훈련 데이터셋이 모델을 한 번 통과하는 주기
learning_rate = 0.0002  ## 학습률 -> 모델 가중치를 업데이트할 때 적용되는 스텝의 크기를 결정
batch_size = 32  ## 배치 크기 -> 모델을 한 번에 학습시키는 데이터의 수

## 이미지 전처리 단계 정의
## Compose: 여러 전처리 단계를 결합하는 객체
transform = transforms.Compose([
    transforms.ToTensor(), ## 이미지를 PyTorch 텐서로 변환 -> 변환 후 이미지의 픽셀 값은 0에서 1 사이의 값으로 스케일링됨
    ## 텐서는 PyTorch에서 데이터를 다루는 기본 단위
    # transforms.Resize((224, 224)), ## 224x224로 이미지 크기 조정
    ## VGG16과 같은 일부 모델은 입력 이미지의 크기가 고정되어 있기 때문에, 모든 입력 이미지를 동일한 크기로 조정해야 함
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ## 이미지 픽셀 값 범위 정규화
    ## 평균과 표준편차는 모두 (0.5, 0.5, 0.5)로 설정 -> RGB 채널 모두에 적용되며, 실제로는 픽셀 값 범위를 [-1, 1]로 조정
])

cifar10_train = datasets.CIFAR10(root='./Data/',train=True,transform=transform,target_transform=None,download=True)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)


## 훈련 데이터셋 로딩
## PyTorch의 datasets와 DataLoader를 사용하여 학습 데이터를 로딩하고 배치 처리를 설정하는 과정
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)
## shuffle=True: 학습 과정에서 데이터셋의 샘플을 무작위로 섞어 오버피팅을 방지하고 학습 과정을 개선


## VGG16 모델 인스턴스 생성 및 device에 할당
model = VGG16(base_dim=64).to(device)
## base_dim: 네트워크의 첫 번째 합성곱 층에서의 출력 차원 수

## 손실 함수 CrossEntropyLoss를 사용 -> 분류 문제에 적합
loss_func = nn.CrossEntropyLoss()

## 최적화 알고리즘으로 Adam을 사용 -> Adam은 자동으로 학습률을 조절하면서 가중치를 업데이트
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

## 학습 과정
for i in range(num_epoch):  ## num_epoch만큼 반복
    ## 배치 처리
    for j, [image, label] in enumerate(train_loader):  ## train_loader에서 배치 사이즈(batch_size) 만큼 데이터(이미지와 레이블)를 가져옴
        x = image.to(device) ## 이미지 데이터를 현재 device로 이동
        y = label.to(device) ## 레이블 데이터를 현재 device로 이동

        optimizer.zero_grad() ## 이전 그래디언트를 초기화 -> 각 배치 처리 때마다 새로운 그래디언트를 계산하기 위함

        ## 예측 및 손실 계산
        output = model.forward(x) ## 모델에 이미지 데이터를 입력하여 예측값(output) 산출
        loss = loss_func(output, y) ## 손실 계산

        ## 역전파와 가중치 업데이트
        loss.backward() ## 손실 함수의 그래디언트를 계산 (역전파)
        optimizer.step() ## 모델의 가중치 업데이트
    
    if i % 2 == 0: ## 에폭이 2의 배수일 때마다 손실값 출력
        print(f'epoch {i} loss: {loss.item()}')
        loss_arr.append(loss.cpu().detach().numpy()) #detach tensor를 gradient 연산에서 분리

## 학습 완료 후 모델의 가중치 저장
torch.save(model.state_dict(), "./train_model")