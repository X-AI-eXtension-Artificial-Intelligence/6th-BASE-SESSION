from data_loader import train_loader  # 'train_loader'를 data_loader.py에서 임포트
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 모델 학습 함수 정의
def train_model(model, train_loader, test_loader, num_epoch, batch_size, learning_rate, device):
    # 손실 함수: CrossEntropyLoss 
    loss_func = nn.CrossEntropyLoss()
    # 옵티마이저: Adam 옵티마이저 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 손실 값들을 저장할 배열
    loss_arr = []
    
    # 에포크(epoch) 수만큼 반복 (전체 데이터셋을 한 번 학습할 때마다 에포크가 1 증가)
    for i in range(num_epoch):
        # 학습 데이터셋을 배치 단위로 가져오기
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device) 
            y_ = label.to(device)  
        
            optimizer.zero_grad()  # 이전에 계산된 기울기 값 초기화 (기울기 누적 방지)
            output = model.forward(x) 
            loss = loss_func(output, y_)  
            loss.backward()  # 손실에 대한 기울기 계산 (역전파)
            optimizer.step()  # 기울기를 이용해 가중치 업데이트

        # 10번째 에포크마다 손실 값을 출력하고, 그래프를 그리기 위한 배열에 추가
        if i % 10 == 0:
            print(loss)  # 현재 에포크의 손실 값 출력
            loss_arr.append(loss.cpu().detach().numpy())  # 손실 값 배열에 추가 (GPU에서 CPU로 이동 후 numpy 배열로 변환)


