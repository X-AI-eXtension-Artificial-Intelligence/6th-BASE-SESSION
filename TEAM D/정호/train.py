import torchvision.datasets as datasets # Pytorch의 Vision 라이브러리 데이터셋 모듈
import torchvision.transforms as transforms # 이미지 전처리 및 변환을 위한 모듈
from torch.utils.data import DataLoader # 데이터를 미니배치로 로딩하기 위한 DataLoader 모듈

from VGG16 import VGG16
import torch 
import torch.nn as nn # PyTorch 모듈 중 인공 신경망 모델을 설계하는데 필요한 함수를 모아둔 모듈

#setiing
batch_size = 100 # 각 반복에서 모델이 학습하는 데이터 샘플 수
learning_rate = 0.0002 # 학습률 : 한번 업데이트 시 조정할 매개변수의 양
num_epoch = 100 # 전체 데이터셋을 100번 반복해서 학습

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')  # GPU 사용 가능 여부 확인 (사용 가능하다면 "cuda:0"(첫번째 GPU), 안되면 cpu 사용용)
print(device)

# 이미지 데이터를 전처리
transforms = transforms.Compose(   
    [transforms.ToTensor(),        # 이미지를 Pytorch 텐서로 변환(numpy -> tensor) 
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]  # 평균(첫번째 인자)과 표준편차(두번째 인자)를 사용하여 각 채널의 픽셀 값을 정규화
)

# 데이터셋 load
cifar10_train = datasets.CIFAR10(root='./Data/', train=True, transform=transforms, target_transform=None, download=True)
cifar10_test = datasets.CIFAR10(root='./Data/', train=False, transform=transforms, target_transform=None, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True) # shuffle=True: 데이터를 무작위로 섞음 -> 모델 일반화 성능 향상
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device) # 모델 정의 # base_dim: 1번째 레이어의 필터 개수(= 출력 채널 수) # to(device): 모델을 지정된 장치(GPU)로 이동
loss_func = nn.CrossEntropyLoss()     # 손실 함수 정의
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer설정 # model.parameters:모델의 학습 가능한 파라미터들을 반환

loss_arr = [] # epoch마다 손실 값을 저장하기 위한 빈 리스트

# 학습 루프
for i in range(num_epoch): 
    # 미니배치에 대한 루프
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device) # 입력 이미지를 지정된 장치(GPU)로 이동
        y = label.to(device) # label을 지정된 장치(GPU)로 이동

        optimizer.zero_grad() # 기울기 초기화: 이전 단계에서 계산된 기울기를 0으로 초기화(매 반복마다 기울기를 새로 계산하기 위함)
        output = model.forward(x)  # 입력(X)를 모델에 전달하여 예측값 output 생성
        loss = loss_func(output,y) # 예측값과 실제 레이블 간의 손실을 계산
        loss.backward()            # 역전파를 수행하여 gradient 계산
        optimizer.step()           # optimizer를 사용하여 모델의 파라미터 업데이트
    
    # 손실 출력 및 저장
    if i%10 == 0 : # 10 epoch마다 한 번씩 현재 손실을 출력
        print(f'epoch {i} loss : ', loss) 
        loss_arr.append(loss.cpu().detach().numpy()) # 손실 값을 numpy 배열로 변환하여 리스트(loss_arr)에 추가  # loss.detach(): 텐서를 gradient 계산에서 분리 # loss.cpu(): 텐서를 CPU 메모리로 이동

# 모델의 학습된 가중치들을 저장
torch.save(model.state_dict(), "./train_model")