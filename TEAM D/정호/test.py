import torch
import torchvision.transforms as transforms

from VGG16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

batch_size = 32

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# CIFAR10 데이터셋 로딩(test)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("./train_model"))   # 훈련된 파라미터 로딩

correct = 0   # 정확히 예측된 데이터 수
total = 0     # 전체 데이터 수

model.eval()  # 모델을 평가 모드로 설정

with torch.no_grad():                                                 # 그라디언트 계산X(메모리 사용량 감소)
    for i, [image, label] in enumerate(test_loader):
        x = image.to(device) 
        y = label.to(device) 

        output = model(x)                                            
        _, output_index = torch.max(output, 1)                        # 예측된 클래스 인덱스

        total += label.size(0)                                        # 테스트 데이터 수 갱신
        correct += (output_index == y).sum().float()                  # 정확한 예측 수 갱신
    
    print("Accuracy of Test Data: {}%".format(100 * correct / total)) # 정확도 출력