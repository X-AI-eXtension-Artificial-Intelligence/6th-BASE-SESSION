import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from vggnet import VGG

#Device 설정 (GPU 사용 가능하면 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU가 있다면 GPU 사용, 없으면 CPU 사용
print(f"Using device: {device}")  # 현재 사용 중인 장치 출력

#데이터 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 픽셀 값을 -1~1 범위로 정규화
])

batch_size = 100
cifar10_test = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)  # 학습 데이터 배치 로드
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)  # 테스트 데이터 배치 로드

# 현재 스크립트가 위치한 디렉토리 찾기
current_dir = os.path.dirname(os.path.realpath(__file__))

# 모델 경로 설정 (현재 디렉토리 기준으로 상대 경로)
model_path = os.path.join(current_dir, "vgg_cifar10.pth")

# 모델 로드
model = VGG(base_dim=64).to(device) #VGG불러오기기
model.load_state_dict(torch.load("vgg_cifar10.pth"))

# 모델 평가
correct = 0  # 맞은 개수를 저장할 변수 초기화
total = 0  # 전체 개수를 저장할 변수 초기화
model.eval()  # 모델을 평가 모드로 전환 (드롭아웃 및 배치 정규화 비활성화)

with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    for images, labels in test_loader:  # 테스트 데이터 로더에서 배치 단위로 이미지와 라벨을 가져옴
        images, labels = images.to(device), labels.to(device)  # 데이터를 GPU 또는 CPU로 이동
        outputs = model(images)  # 모델에 입력 데이터를 넣고 예측값 출력
        _, predicted = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스의 인덱스를 예측값으로 설정
        total += labels.size(0)  # 전체 샘플 수 누적
        correct += (predicted == labels).sum().item()  # 맞춘 개수 누적

    accuracy = 100 * correct / total  # 정확도 계산 (정확히 예측한 샘플 수 / 전체 샘플 수 * 100)
    print(f"Accuracy on Test Data: {accuracy:.2f}%")  # 정확도 출력
