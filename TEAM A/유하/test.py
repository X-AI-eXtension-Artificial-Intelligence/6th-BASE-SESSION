import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG

# hyperparameter
batch_size = 100
learning_rate = 0.0002
save_path = "vgg_model.pth"

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 저장된 모델 불러오기 
try:
    model.load_state_dict(torch.load(save_path))
    print(f"저장된 모델 {save_path} 불러오기 성공!")
except FileNotFoundError:
    print(f"저장된 모델 {save_path}을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Transform 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 TEST 데이터 정의
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, target_transform=None, download=True)
# DataLoader 정의
test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle=False)

# test
correct = 0  # 맞춘 개수를 저장하는 변수
total = 0  # 전체 샘플 개수를 저장하는 변수

# .eval() 호출 -> 모델이 평가 모드로 전환됨 (dropout & batchnorm 동작 방식이 달라짐)
model.eval()

# 기울기 업데이트 비활성화 (test 과정에서는 역전파가 필요 없기 때문에 실행해줘야 함) -> 연산 속도도 빨라지고, 메모리 사용량도 줄어둚
with torch.no_grad():
    # test_loader에서 batch 단위로 데이터 불러옴
    for image,label in test_loader:
        
        # GPU / CPU 로 이동시킴
        x = image.to(device)  # 입력 데이터 : (100, 3, 32, 32)
        y= label.to(device)  # 정답 데이터 : (100, )

        # output 생성
        output = model.forward(x)
        # 각 샘플에서 가장 높은 점수를 가진 클래스 인덱스 저장 -> 예측 값으로 선택하기 위해
        _,output_index = torch.max(output,1)

        # 정답 개수 & 전체 개수 업데이트
        total += label.size(0)  # 전체 개수 += 배치 크기
        correct += (output_index == y).sum().float()  # 맞춘 개수 += 예측과 정답이 같은 개수
    
    # 정확도 도출 = (맞춘 개수 / 전체 개수) * 100
    print("Accuracy of Test Data: {}%".format(100*correct/total))