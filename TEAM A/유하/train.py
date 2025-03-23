import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import trange
from model import VGG

# hyperparameter 설정
batch_size = 100  # 한 번의 학습에서 사용할 데이터 개수 
learning_rate = 0.0002  # 모델을 얼마나 빠르게 학습할지 (너무 크면, 최적 값 도달 X / 너무 작으면, 학습 속도가 느려지고 오버피팅 가능성 증가)
num_epoch = 100  # 전체 데이터를 몇 번 반복하여 학습할 것인지
save_path = "vgg_model.pth"  # 모델 가중치 저정할 파일 경로

# device 설정 -> 가능하면 GPU, 없으면 CPU로 학습 진행하는 코드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화 (VGG 클래스를 객체로 생성)
## 인자 : 1번째 합성곱 레이어 채널 수 -> 64로 설정
## .to(device) 안 하면 GPU에서 학습이 안 됨
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()  # 교차 엔트로피 : 다중 클래스 분류 문제에서 사용되는 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 가중치 업데이트 (인자 : 모델의 학습 가능한 모든 파라미터 전달, 학습률)

# Adam optimizer : SGD보다 빠르고, 학습률 조정이 자동화됨 / 대부분의 딥러닝 모델에서 기본적으로 사용됨

# Transform 정의 -> 데이터 전처리
## transforms.Compose : 여러 개의 transform 작업을 묶어서 실행
transform = transforms.Compose(
    [transforms.ToTensor(),  # 이미지 데이터를 Pytorch Tensor로 변환
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 데이터 정규화 (각 채널을 평균 0, 표준편차 1로 변환) / [0,1] -> [-1,1]

# CIFAR10 TRAIN 데이터셋 로드
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, target_transform=None, download=True)
# DataLoader 정의
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)

# train
loss_arr = []  # 손실 값 저장용 리스트
for i in trange(num_epoch):  # num_epoch 만큼 반복 학습
    for j,[image,label] in enumerate(train_loader):  # train_loader에서 배치 단위로 데이터를 불러 옴
        x = image.to(device)  # 입력 데이터 : (100, 3, 32, 32)
        y_= label.to(device)  # 정답 데이터 : (100, )
        
        optimizer.zero_grad()  # 이전 배치에서 계산된 기울기 초기화
        output = model.forward(x)  # 순전파 / output 크기 : (batch_size, num_classes) = (100,10)
        loss = loss_func(output,y_)  # 손실 계산
        loss.backward()  # 역전파 -> 기울기 계
        optimizer.step()  # 가중치 업데이

    if i % 10 ==0:  # 10 epoch마다 손실값 출력 및 저장 (-> 손실 값이 점점 작아지면 학습이 잘 되고 있음을 의미함)
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())

# 학습된 모델 저장 
torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료: {save_path}")