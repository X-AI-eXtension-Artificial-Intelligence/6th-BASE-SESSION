import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VGG16 import VGG16
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

# 이미지 데이터를 전처리
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

# train=False : 테스트용 데이터셋 load
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model')) # 훈련된 모델의 파라미터를 load

# eval
correct = 0   # 정확히 예측된 데이터 수
total = 0     # 전체 데이터 수

model.eval()  # 모델을 평가 모드로 설정

with torch.no_grad(): # gradient 계산 비활성화: 평가 시에는 모델을 업데이트하지 않기 때문(메모리 사용량 줄이고 계산 속도 향상)
    for i,[image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)  # forward -> 예측 결과 계산 # output의 크기 = 배치 크기 x 클래스 수
        
        _,output_index = torch.max(output,1)  # output 텐서의 두 번쨰 차원(1)을 따라 최대값과 그 인덱스 반환
                                              # 최대값은 버리고(_), 인덱스값만 output_index에 할당

        total += label.size(0) # 현재 배치의 크기를 total 변수에 더하여 전체 샘플 수 업데이트
        correct += (output_index==y).sum().float() # 정확하게 예측된 샘플의 수 업데이트

    print("Accuracy of Test DataL {}%".format(100*correct/total))  # 정확도 계산