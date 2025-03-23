# test.py
# 이 파일은 학습된 모델을 불러와서 테스트 데이터셋에 대해 평가하는 역할을 수행한다.
# 학습 시 저장된 모델 파라미터를 불러와 테스트 데이터에 대해 정확도를 측정한다.

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# model.py에 정의된 VGG 모델 클래스를 불러온다.
from model import VGG

# -----------------------------------------------------------------------------
# 1. 테스트 데이터셋 전처리 및 DataLoader 설정
# -----------------------------------------------------------------------------
# 학습 시와 동일하게 이미지를 텐서로 변환하고 정규화하는 transform을 정의한다.
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5),  # 각 채널의 평균값
                         (0.5, 0.5, 0.5))  # 각 채널의 표준편차
])

# CIFAR10 테스트 데이터셋을 다운로드하고 transform을 적용한다.
test_dataset = datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)
# DataLoader를 사용하여 테스트 데이터를 배치 단위로 불러온다. (배치 크기는 100)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# -----------------------------------------------------------------------------
# 2. 디바이스 설정 및 모델 인스턴스 생성
# -----------------------------------------------------------------------------
# GPU 사용 가능 여부에 따라 GPU 또는 CPU를 사용한다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 학습 때와 동일한 구조의 VGG 모델을 생성한다.
model = VGG(base_dim=64, num_classes=10).to(device)

# -----------------------------------------------------------------------------
# 3. 저장된 모델 파라미터 불러오기
# -----------------------------------------------------------------------------
# train.py에서 저장한 'vgg_model.pth' 파일로부터 모델의 파라미터(state_dict)를 불러온다.
# map_location=device를 사용하여 현재 디바이스에 맞게 모델 파라미터를 불러온다.
model.load_state_dict(torch.load('vgg_model.pth', map_location=device))
# 모델을 평가 모드로 전환하여 dropout 등 학습 중에만 필요한 기능을 비활성화한다.
model.eval()

# -----------------------------------------------------------------------------
# 4. 테스트 데이터셋에 대한 평가 진행
# -----------------------------------------------------------------------------
# 정확하게 분류된 이미지 수와 전체 이미지 수를 기록할 변수를 초기화한다.
correct = 0
total = 0

# 평가 시에는 gradient 계산이 필요 없으므로 torch.no_grad()로 감싼다.
with torch.no_grad():
    # DataLoader를 통해 테스트 데이터를 배치 단위로 가져온다.
    for image, label in test_loader:
        # 이미지를 지정된 디바이스(GPU 또는 CPU)로 이동시킨다.
        inputs = image.to(device)
        targets = label.to(device)
        # 모델을 통해 예측 결과를 계산한다.
        outputs = model(inputs)
        # torch.max를 사용하여 출력 값 중 가장 큰 값을 가진 인덱스를 예측 결과로 선택한다.
        _, predicted = torch.max(outputs, 1)
        # 배치 내 총 이미지 수를 누적한다.
        total += targets.size(0)
        # 예측값과 실제 라벨이 일치하는 경우를 카운트하여 누적한다.
        correct += (predicted == targets).sum().item()

# 전체 테스트 데이터셋에 대한 정확도를 계산한다.
accuracy = 100 * correct / total
# 최종적으로 테스트 데이터셋에 대한 정확도를 출력한다.
print("테스트 데이터 정확도: {:.2f}%".format(accuracy))
