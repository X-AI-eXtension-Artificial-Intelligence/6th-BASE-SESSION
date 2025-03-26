
import torch  # 텐서 연산, GPU 연산, 자동 미분 등의 기능

import torch.nn as nn  # nn.Linear, nn.Conv2d, nn.ReLU 등을 사용하여 뉴럴 네트워크를 구성.
                       # fc = nn.Linear(10, 5)  # 입력 노드 10개 → 출력 노드 5개 (완전연결층)


import torch.optim as optim  # 최적화 알고리즘을 제공하는 라이브러리
                             # optim.SGD, optim.Adam, optim.RMSprop 등등

import torchvision  # 컴퓨터 비전 관련 데이터셋, 모델, 변환(transform) 등을 포함하는 라이브러리
                    # ResNet, VGG, MobileNet 등 사전 학습된 모델, 
                    # torchvision.datasets.MNIST, torchvision.datasets.CIFAR10, torchvision.datasets.ImageNet

import torchvision.transforms as transforms  # 데이터의 전처리를 위한 변환(transform) 함수
                                             # 데이터 증강(augmentation) 및 정규화(normalization) 등의 기능 포함.

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
# functions to show an image

# transforms.Compose하면 모듈화처럼 순차적으로 전처리 구현 가능
def data_transform(train = True): 
    transform = transforms.Compose(  # 여러 개의 이미지 변환(transform) 연산을 순차적으로 적용할 수 있도록 묶어주는 함수
    [transforms.ToTensor(),  # 이미지 파일 Tensor 형태로 바꾸고 0~1 범위로 자동 정규화
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]  # 평균, 표준편차로 정규화. RGB 3채널 이미지 이므로 각각 지정 - 입력 데이터 분포 일정하게 유지하면 학습 안정성 및 일반화 능력 향상
    )

    #torchvision 내장 CIFAR10 Dataset 활용(target_transform - 레이블은 변환 없음)
    cifar10_dataset = torchvision.datasets.CIFAR10(root = "./cifar10", 
                                                   train = train,  # T/F로 로드할 훈련/테스트 데이터 결정 
                                                   transform=transform,  # 정규화 및 텐서 변형 적용 
                                                   target_transform=None,  #  라벨 변환 없음. 기본적으로 CIFAR-10의 라벨(클래스 번호)은 0~9 범위의 정수
                                                   download = True)  # 데이터 없으면 자동으로 다운로드 
    return cifar10_dataset

# 클래스 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





# 데이터셋 이미지 시각화
def imshow(img):
    img = img / 2 + 0.5 #정규화 풀고 다시 0~1 범위로
    npimg = img.numpy() #image numpy 형태로 변형
    plt.imshow(np.transpose(npimg, (1,2,0)))#파이토치 텐서 C(채널),H,W 순서라 -> H,W,C 형태로 변형
    plt.savefig('CIFAR10_Image.png') #이미지 저장
    plt.show()
    plt.close()

## 데이터 로더에서 무작위로 이미지 가져와서 격자 형태로 시각화
def random_viualize(data_loader, batch_size):
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    #make_grid로 여러 이미지 grid 형태로 묶어서 출력
    imshow(torchvision.utils.make_grid(images))

    #배치 만큼의 이미지 클래스 라벨 텍스트로 변환해서 출력
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))





















