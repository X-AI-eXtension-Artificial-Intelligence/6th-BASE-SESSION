import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환 (PIL 이미지를 PyTorch 텐서로 변환)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지 정규화
])

# CIFAR-10 학습 데이터셋 불러오기
cifar10_train = datasets.CIFAR10(
    root="../Data/",  # 데이터셋을 저장할 경로
    train=True,  # 학습용 데이터셋
    transform=transform,  # 정의한 변환(transform)을 적용
    download=True  # 데이터셋이 없다면 자동으로 다운로드
)

# CIFAR-10 테스트 데이터셋 불러오기
cifar10_test = datasets.CIFAR10(
    root="../Data/",  # 데이터셋을 저장할 경로
    train=False,  # 테스트용 데이터셋
    transform=transform,  # 정의한 변환(transform)을 적용
    download=True  # 데이터셋이 없다면 자동으로 다운로드
)

# DataLoader 정의 (배치 크기 설정 및 셔플링)
train_loader = DataLoader(
    cifar10_train,  # 학습용 데이터셋
    batch_size=100,  # 한 번에 100개의 데이터를 배치로 로드
    shuffle=True,  # 데이터 순서를 무작위로 섞음 (학습 시 유용)
    num_workers=2  # 데이터 로딩을 위한 프로세스 수 (병렬 처리)
)

test_loader = DataLoader(
    cifar10_test,  # 테스트용 데이터셋
    batch_size=100,  # 한 번에 100개의 데이터를 배치로 로드
    shuffle=False,  # 테스트 데이터는 셔플하지 않음
    num_workers=2  # 데이터 로딩을 위한 프로세스 수
)

# CIFAR-10 클래스 목록 (각 클래스에 대응하는 이름)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
