import torch
import torch.nn as nn
from tqdm import trange

batch_size = 100
learning_rate = 0.0002
num_epoch = 100

loss_arr = []
for i in trange(num_epoch):  # num_epoch 만큼 반복 
    for j,[image,label] in enumerate(train_loader): 
        x = image.to(device)
        y_= label.to(device)

        optimizer.zero_grad()  # 기울기 0으로 초기화 
        output = model.forward(x)  # 입력 이미지를 모델에 전달하여 예측 값을 계산
        loss = loss_func(output,y_)
        loss.backward()  # 기울기 계산 
        optimizer.step() # 가중치 업데이트 

    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())  # 손실 값을 기록하여 후속 분석에 활용하기 위함
