import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import data_transform
from model import VGG16

batch_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #CUDA GPU 활용

#train=False로 test set 설정
test_loader = DataLoader(data_transform(train=False), batch_size = batch_size, shuffle = True, num_workers = 2)

# model 정의
model = VGG16(base_dim=64).to(device) #위의 설계 모델(기본 차원 64) -> GPU에 올리기
model.load_state_dict(torch.load("Best_VGG16_model.pth")) #가중치 load

#모델 평가 모드 전환
model.eval()

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss() #분류 문제이기 때문에 크로스엔트로피 손실 함수 지정

#정확도 계산을 위한 전체 개수, 맞은 개수 변수 생성
correct = 0
total = 0


# 테스트
with torch.no_grad(): #gradient 계산 안함(학습 X)
    for image,label in test_loader:

        #테스트셋도 동일하게 Image, label 별로 GPU에 얹기
        x = image.to(device) 
        y = label.to(device)

        output = model.forward(x) #순전파 수행
        _, output_index = torch.max(output,1) #예측 확률중 가장 큰 것과 그 인덱스 반환

        total += label.size(0) #전체 배치 샘플 수
        correct += (output_index == y).sum().float() #맞은 개수 세기 

print(total)
print(correct)
print("Accuracy of Test Data: {}%".format(100*correct/total)) #최종 예측 정확도 출력