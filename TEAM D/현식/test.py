import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG 

batch_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

# train=False : 테스트용 데이터셋 로드
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

model = VGG(base_dim=64).to(device)
model.load_state_dict(torch.load('./models/VGG16_100.pth')) # 학습한 모델 가중치 로드

correct = 0
total = 0

# 학습 중에만 사용하는 Dropout, Batch Normalization을 비활성화
# Batch Normalization은 학습 중에는 현재 배치의 통계(평균과 분산)을 사용
# 하지만, 평가 중에는 학습 중 추적된 이동 평균과 분산을 사용하여 데이터 정규화
model.eval() 

# 평가 과정에서는 gradient 계산이 필요 없으므로 비활성화하여 메모리 사용량과 계산 속도 최적화
with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        # output 텐서에서 가장 큰 값의 인덱스를 추출 (모델이 예측한 class label)
        # dim=1 : 각 샘플의 class 차원을 기준으로 최대값을 찾음
        # _ : 최대값을 저장하지 않고, 필요 없는 값 무시
        _, output_index = torch.max(output,1) 

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))

# Accuracy of Test Data : 79.68999481201172%