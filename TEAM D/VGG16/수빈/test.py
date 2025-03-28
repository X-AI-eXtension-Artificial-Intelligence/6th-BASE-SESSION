import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import VGG16

# 장치 설정 - GPU 사용 가능하면 GPU, 없으면 CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 배치 사이즈 설정
batch_size = 100  # 한 번에 100개의 이미지 처리

# 이미지 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 평균 0.5, 표준편차 0.5 기준 정규화
])

# 데이터셋 로드
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, download=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# 모델 로드
model = VGG16(base_dim=64, num_classes=10).to(device)  # num_classes=10 추가
model.load_state_dict(torch.load('./models/VGG16_100.pth', map_location=device))

# 변수 초기화
correct = 0  # 정답 개수 저장 변수
total = 0  # 전체 데이터 개수 저장 변수

# 모델을 평가 모드로 전환 (Dropout, BatchNorm 비활성화)
model.eval()

# 모델 추론 (gradient 계산 안 함 → 메모리 절약 & 속도 향상)
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):  
        x = image.to(device)  
        y = label.to(device)  

        output = model(x)  # 모델 예측 수행
        _, output_index = torch.max(output, 1)  # 가장 높은 확률값을 가진 클래스의 인덱스 반환
        total += label.size(0)  # 총 데이터 수 업데이트
        correct += (output_index == y).sum().item()  # 정답 개수 업데이트

# 정확도 출력
accuracy = 100 * correct / total
print(f"Accuracy of Test Data: {accuracy:.2f}%")
