import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import data_transform
from dataset import random_visualize
from model import VGG16
from tqdm import trange # 모델 학습과정 tqdm 활용해서 range안에 넣고 루프 진행상황 보기

# 배치 사이즈, 학습률, 에포크 지정
batch_size = 100
learning_rate = 0.00005
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #CUDA GPU 활용


# DataLoader로 train, test set 준비, 순서 섞기
train_loader = DataLoader(data_transform(train=True), batch_size = batch_size, shuffle = True, num_workers = 2) #함수에서 이미 FashionMnist load -> data_transform 함수

# train set 시각화
random_visualize(train_loader)

# model 정의
model = VGG16(base_dim=64).to(device) #위의 설계 모델(기본 차원 64) -> GPU에 올리기

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() #분류 문제이기 때문에 크로스엔트로피 손실 함수 지정
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #Adam Optimizer 활용


#학습
loss_arr = [] #loss 담아줄 array 생성

best_loss = float('inf')  # 초기값을 무한대로 설정

for i in trange(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        # 10번째 배치마다 loss 출력 후 array에 저장
        if i % 10 == 0:
            print(f"Epoch [{i}/{num_epoch}], Step [{j}], Loss: {loss.item():.4f}")
            loss_arr.append(loss.cpu().detach().numpy())

        # 현재 loss가 best_loss보다 작으면 모델 저장
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "Best_VGG16_model.pth")  # 가장 좋은 가중치 저장
            print(f"Best model saved with loss: {best_loss:.4f}")

# loss curve 그리기
plt.plot(loss_arr)
plt.savefig('FashionMNIST_VGG16_Loss_curve.png')
plt.show()
