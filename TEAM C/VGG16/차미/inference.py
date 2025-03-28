import torch
import torchvision.transforms as transforms
from PIL import Image

from vgg16 import VGG16
from torch.utils.data import DataLoader
from torchvision import datasets

## GPU 사용 가능 여부에 따라 device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1

## 데이터에 적용할 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 원하는 크기로 조정
    transforms.ToTensor(), ## 이미지를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ## 이미지 픽셀 값을 정규화
])

cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, download=True)

## 데이터셋 로딩
image_path = './dog.jpeg'
image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  # 배치 차원 추가 (1, C, H, W)


## 모델 인스턴스 생성 및 저장된 가중치 로드
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("./train_model"))

model.eval() ## 모델을 평가 모드로 설정

with torch.no_grad(): ## 그래디언트 계산을 비활성화하여 메모리 사용량 줄이고 계산 속도 향상
    image = image.to(device)
    output = model(image) ## 모델에 이미지 데이터를 입력하여 예측값 산출
    _, predicted = torch.max(output, 1) ## 예측된 클래스 인덱스
    predicted_class = cifar10_test.classes[predicted.item()]
    print("Predicted class:", predicted_class)
    print("Predicted class index:", predicted.item())