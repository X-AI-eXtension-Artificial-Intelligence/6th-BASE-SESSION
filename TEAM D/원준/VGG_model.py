import torch
import torch.nn as nn
from tqdm import trange  # 학습 진행 과정 시각적으로 보여주는 라이브러리

# 학습률 설정
learning_rate = 0.001

# 두 개의 컨볼루션 레이어를 포함하는 블록 정의
def conv_2_block(in_dim, out_dim):  # in_dim: 입력 이미지 또는 특징 맵의 채널 수 out_dim: 출력 특징 맵의 채널 수
    model = nn.Sequential(         #  nn.Sequential은 파이토치에서 여러 층을 순차적으로 쌓을 때 사용하는 컨테이너
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 컨볼루션 레이어
        nn.ReLU(), 
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 컨볼루션 레이어
        nn.ReLU(), 
        nn.MaxPool2d(2, 2)  # kernel_size=2 * 2 , stride=2
    )
    return model

# 세 개의 컨볼루션 레이어를 포함하는 블록 정의
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)  
    )
    return model

# VGG 모델 정의
class VGG(nn.Module): # nn.Module은 PyTorch에서 모든 신경망 모델의 기본 클래스
    def __init__(self, base_dim, num_classes=10):  # 생성자 자동 실행, base_dim 모델 기본 채널 크기 설정, 분류할 클래스 개수
        super(VGG, self).__init__()   # nn.Module의 기능을 상속받기 위해 super() 호출 
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 입력 채널 3(RGB), 출력 채널 base_dim,  위에서 정의한 것 
            conv_2_block(base_dim, 2 * base_dim),  # 채널 수 2배로 증가
            conv_3_block(2 * base_dim, 4 * base_dim),
            conv_3_block(4 * base_dim, 8 * base_dim),
            conv_3_block(8 * base_dim, 8 * base_dim),
        )
        self.fc_layer = nn.Sequential(  # 완전 연결 계층 
            nn.Linear(8 * base_dim * 1 * 1, 4096),  # CIFAR-10 입력 크기에 맞게 설정
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000), # 4096차원 벡터 입력받아 1000차원 벡터로 변환
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),  # 클래스 예측 
        )

    def forward(self, x): # 모델이 데이터를 처리하는 방법을 정의하는 함수
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # view 크기 변경 
                                    # x.size(0) 배치 사이즈  빼고 나머지 납작
                                    # 함수텐서를 펼쳐서 Fully Connected Layer에 입력 일렬로 배열열
        x = self.fc_layer(x)
        return x

# CUDA 사용 여부 확인 후 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 모델 인스턴스화 및 디바이스로 이동
model = VGG(base_dim=64).to(device) # 첫 번째 컨볼루션 레이어가 출력하는 채널 수가 64

# 손실 함수 및 최적화 기법 정의
loss_func = nn.CrossEntropyLoss()  # 다중 분류 문제에 적합한 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam Optimizer 사용

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
