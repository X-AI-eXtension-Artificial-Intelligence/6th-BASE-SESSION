import torch
import torchvision.transforms as transforms

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

## GPU 사용 가능 여부에 따라 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

## 테스트 데이터에 적용할 이미지 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(), ## 이미지를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ## 이미지 픽셀 값을 정규화
])

## 테스트 데이터셋 로딩
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

## 모델 인스턴스 생성 및 저장된 가중치 로드
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("./train_model"))

# eval
correct = 0 ## 정확히 예측된 데이터 수
total = 0 ## 전체 데이터 수

model.eval() ## 모델을 평가 모드로 설정

with torch.no_grad(): ## 그래디언트 계산을 비활성화하여 메모리 사용량 줄이고 계산 속도 향상
    for i, [image, label] in enumerate(test_loader):
        x = image.to(device) ## 테스트 이미지를 현재 device로 이동
        y = label.to(device) ## 레이블 데이터를 현재 device로 이동

        output = model(x) ## 모델에 이미지 데이터를 입력하여 예측값 산출
        _, output_index = torch.max(output, 1) ## 예측된 클래스 인덱스

        total += label.size(0) ## 테스트 데이터 수 갱신
        correct += (output_index == y).sum().float() ## 정확한 예측 수 갱신
    
    print("Accuracy of Test Data: {}%".format(100 * correct / total)) ## 정확도 출력