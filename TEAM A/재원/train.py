import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import dataset_loader
from model import VGG19
from tqdm import trange # 모델 학습과정 tqdm 활용해서 range안에 넣고 루프 진행상황 보기

# 학습률, 에포크 지정
learning_rate = 0.00005
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #CUDA GPU 활용

# DataLoader로 train set 준비
train_loader = dataset_loader("train", 32, True)

# model 정의
model = VGG19(base_dim=64).to(device) #위의 설계 모델(기본 차원 64) -> GPU에 올리기

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() #분류 문제이기 때문에 크로스엔트로피 손실 함수 지정
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #Adam Optimizer 활용

# 한 배치의 라벨 값들을 확인하는 예시
for j, batch in enumerate(train_loader):
    labels = batch["labels"]
    print("배치의 라벨 값:", labels)
    print("라벨 최소값:", labels.min().item(), "라벨 최대값:", labels.max().item())
    break

# 학습 (배치 key값 -> [image, labels]) -> batch로 담아와서 딕셔너리 언패킹 필요
loss_arr = [] #loss 담아줄 array 생성
for i in trange(num_epoch): #100 epoch 학습
    for j,batch in enumerate(train_loader): 
        
        image = batch['image']
        label = batch['labels']

        #GPU에 이미지랑 Label 얹기
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad() #이전 gradient 초기화
        output = model.forward(x) #순전파
        loss = loss_func(output, y_) #손실함수 계산
        loss.backward() #역전파
        optimizer.step() #가중치 업데이트

        # 3번째 배치마다 loss 출력 후 array에 저장
        if i % 3 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())


# loss curve 그리기
plt.plot(loss_arr)

# loss curve 그래프 이미지 저장
plt.savefig('Pokemon_VGG19_Loss_curve.png')
plt.show()

# model 저장
torch.save(model.state_dict(), "VGG19_model.pth")