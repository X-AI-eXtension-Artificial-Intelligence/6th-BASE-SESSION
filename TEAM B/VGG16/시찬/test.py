# torch 및 torchvision 관련 모듈을 임포트하여 평가에 필요한 도구들을 사용
import torch  
import torchvision.datasets as datasets  # 데이터셋 로드에 사용
import torchvision.transforms as transforms  # 이미지 전처리 도구
# model.py에서 정의한 VGG 모델 클래스를 불러온다.
from model import VGG  

# 평가 단계에서는 데이터 증강 없이 정규화만 적용하므로, ToTensor와 Normalize만 수행
transform = transforms.Compose([
    # 이미지를 파이토치 텐서로 변환
    transforms.ToTensor(),
    # CIFAR10의 각 채널을 정규화
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 테스트 데이터셋을 다운로드하고, 위에서 정의한 transform을 적용
test_dataset = datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)
# DataLoader를 사용하여 테스트 데이터를 배치 단위로 불러온다. num_workers=4로 로딩 속도를 높임임
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, num_workers=4)

# GPU 사용 가능 여부에 따라 GPU 또는 CPU를 사용하도록 디바이스를 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 학습 시와 동일한 VGG 모델 구조로 모델 인스턴스를 생성하고, 디바이스로 이동시킴킴
model = VGG(base_dim=64, num_classes=10).to(device)

# train.py에서 저장한 모델 파라미터를 불러오고, 현재 디바이스에 맞게 로드
model.load_state_dict(torch.load('vgg_model.pth', map_location=device))
# 모델을 평가 모드로 전환하여 dropout 등 학습 전용 기능을 비활성화
model.eval()

# 테스트 데이터셋에서 올바르게 분류한 이미지의 개수를 누적할 변수 초기화
correct = 0
# 테스트 데이터셋의 총 이미지 개수를 누적할 변수 초기화
total = 0

# 평가 단계에서는 gradient 계산이 필요 없으므로 torch.no_grad()를 사용하여 메모리 사용량을 줄임임
with torch.no_grad():
    # DataLoader를 통해 배치 단위로 테스트 데이터를 순회
    for images, labels in test_loader:
        # 이미지와 라벨 데이터를 디바이스(GPU 또는 CPU)로 이동시킴킴
        images = images.to(device)
        labels = labels.to(device)
        # 모델의 forward 메소드를 호출하여 예측 결과를 산출
        outputs = model(images)
        # torch.max를 사용하여 각 배치에서 가장 큰 확률의 인덱스를 예측 결과로 선택
        _, predicted = torch.max(outputs, 1)
        # 현재 배치의 총 이미지 수를 누적
        total += labels.size(0)
        # 예측 결과와 실제 라벨이 일치하는 경우를 누적
        correct += (predicted == labels).sum().item()

# 전체 테스트 데이터셋에 대한 정확도를 계산
accuracy = 100 * correct / total
# 최종 테스트 정확도를 출력
print("테스트 데이터 정확도: {:.2f}%".format(accuracy))
