import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG 

from tqdm import trange

loss_arr = []  # 에포크별 손실 값을 저장할 리스트
for i in trange(num_epoch):  # num_epoch만큼 학습 반복
    for j, [image, label] in enumerate(train_loader):  # 미니배치 단위로 데이터 로드
        x = image.to(device)  # 입력 이미지 GPU/CPU로 전송
        y_ = label.to(device)  # 정답 라벨 GPU/CPU로 전송

        optimizer.zero_grad()  # 기존 기울기(gradient) 초기화
        output = model.forward(x)  # 모델의 순전파(forward propagation) 수행
        loss = loss_func(output, y_)  # 손실(loss) 계산
        loss.backward()  # 역전파(backpropagation) 수행 (기울기 계산)
        optimizer.step()  # 최적화 수행 (가중치 업데이트)

    if i % 10 == 0:  # 10번째 에포크마다 손실 출력 및 저장
        print(loss)  # 현재 에포크의 마지막 배치 손실 출력
        loss_arr.append(loss.cpu().detach().numpy())  # 손실 값을 리스트에 저장

#test  결과

correct = 0
total = 0

model.eval()

# 인퍼런스 모드를 위해 no_grad 해줍니다.
with torch.no_grad():
    # 테스트로더에서 이미지와 정답을 불러옵니다.
    for image,label in test_loader:

        # 두 데이터 모두 장치에 올립니다.
        x = image.to(device)
        y= label.to(device)

        # 모델에 데이터를 넣고 결과값을 얻습니다.
        output = model.forward(x)
        _,output_index = torch.max(output,1)


        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        correct += (output_index == y).sum().float()

    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))
