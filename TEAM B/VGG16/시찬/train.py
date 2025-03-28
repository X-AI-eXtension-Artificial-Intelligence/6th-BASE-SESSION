# torch 및 torch 관련 모듈을 임포트하여 학습에 필요한 도구들을 사용
import torch  
import torch.optim as optim  # 옵티마이저 관련 모듈 임포트
import torch.nn as nn  # 손실함수 및 신경망 구성 관련 모듈 임포트
# torchvision을 사용하여 데이터셋 및 전처리 도구들을 임포트
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
# matplotlib.pyplot을 임포트하여 학습 과정 시각화에 사용
import matplotlib.pyplot as plt  
# numpy를 임포트하여 수치 계산에 사용
import numpy as np  
# tqdm의 trange를 임포트하여 진행 상황을 표시하는 progress bar를 사용
from tqdm import trange  
# model.py에 정의된 VGG 모델 클래스를 불러옴옴
from model import VGG  

# 데이터 증강과 정규화를 포함하는 전처리 과정을 정의
transform = transforms.Compose([
    # CIFAR10 이미지 크기에 맞춰 32x32 크기를 유지하며 padding을 추가하여 랜덤 크롭 수행
    transforms.RandomCrop(32, padding=4),
    # 좌우 반전을 랜덤하게 수행하여 데이터 다양성을 높임임
    transforms.RandomHorizontalFlip(),
    # 이미지를 파이토치 텐서로 변환
    transforms.ToTensor(),
    # 정규화: 각 채널을 평균 0.5, 표준편차 0.5로 정규화
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 학습 데이터셋을 다운로드하고 위에서 정의한 transform을 적용
train_dataset = datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
# DataLoader를 사용하여 데이터를 배치 단위로 불러오며, num_workers=4로 데이터 로딩 속도를 향상
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=4)

# GPU 사용 가능 여부에 따라 GPU 혹은 CPU를 선택하여 디바이스를 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# VGG 모델을 base_dim=64, CIFAR10의 클래스 수 10으로 생성하고 선택한 디바이스로 이동
model = VGG(base_dim=64, num_classes=10).to(device)

# CrossEntropyLoss를 손실 함수로 사용하여 분류 문제에 적합하도록 설정
loss_func = nn.CrossEntropyLoss()
# Adam 옵티마이저를 사용하여 모델의 파라미터를 업데이트하며, 학습률은 0.0002로 설정
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# 학습률 스케줄러를 사용하여 30 에폭마다 학습률을 0.1배로 감소
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Mixed Precision Training을 위해 GradScaler를 생성 이는 FP16 연산을 안정적으로 수행하도록 도움움
scaler = torch.cuda.amp.GradScaler()

# 총 학습 에폭 수를 100으로 설정
num_epoch = 100
# 에폭마다의 평균 손실 값을 저장할 리스트를 초기화
loss_arr = []

# trange를 사용하여 학습 에폭마다 진행 상황을 표시
for epoch in trange(num_epoch, desc="학습 진행"):
    # 모델을 학습 모드로 전환하여 dropout 등 학습에 필요한 기능을 활성화
    model.train()
    # 현재 에폭의 총 손실 값을 누적하기 위한 변수 초기화
    running_loss = 0.0
    # DataLoader를 통해 미니 배치 단위로 데이터를 순회
    for images, labels in train_loader:
        # 이미지와 라벨 데이터를 선택한 디바이스(GPU 또는 CPU)로 이동
        images = images.to(device)
        labels = labels.to(device)
        
        # 옵티마이저의 기울기를 0으로 초기화하여 이전 배치의 기울기가 누적되는 것을 방지
        optimizer.zero_grad()
        # 자동 혼합 정밀도(autocast)를 적용하여 FP16 연산을 수행하고, 계산 속도를 향상
        with torch.cuda.amp.autocast():
            # 모델의 forward 메소드를 호출하여 예측 결과를 산출
            outputs = model(images)
            # 예측 결과와 실제 라벨을 비교하여 손실값을 계산
            loss = loss_func(outputs, labels)
        # 계산된 손실값을 바탕으로 역전파 수행 (GradScaler를 사용하여 수치 안정성을 유지)
        scaler.scale(loss).backward()
        # 옵티마이저의 파라미터를 업데이트
        scaler.step(optimizer)
        # GradScaler의 상태를 업데이트
        scaler.update()
        
        # 해당 배치의 손실값을 누적
        running_loss += loss.item()
    # 한 에폭이 끝난 후 학습률 스케줄러를 통해 학습률을 조정
    scheduler.step()
    
    # 10 에폭마다 평균 손실값을 출력하고 loss_arr에 저장
    if (epoch + 1) % 10 == 0:
        avg_loss = running_loss / len(train_loader)
        print("에폭 [{}/{}], 평균 손실값: {:.4f}".format(epoch+1, num_epoch, avg_loss))
        loss_arr.append(avg_loss)

# 학습 과정의 손실값 변화를 시각화하기 위해 새로운 figure를 생성
plt.figure()
# x축: 에폭 수, y축: 평균 손실값을 plot
plt.plot(np.arange(len(loss_arr)) * 10, loss_arr, marker='o')
plt.title("학습 손실값 변화")
plt.xlabel("에폭")
plt.ylabel("평균 손실값")
plt.grid(True)  # 그래프에 격자선을 추가하여 가독성을 높임
# 시각화한 그래프를 화면에 출력
plt.show()

# 학습이 완료된 모델의 파라미터(state_dict)를 'vgg_model.pth' 파일로 저장
torch.save(model.state_dict(), 'vgg_model.pth')
