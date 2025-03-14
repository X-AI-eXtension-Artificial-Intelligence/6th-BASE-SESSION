import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG 


batch_size = 100
learning_rate = 0.0005
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transforms.Compose : 여러 데이터 전처리 단계를 하나로 묶음
# transforms.ToTensor : 이미지를 PyTorch 텐서로 변환 (PyTorch 모델은 텐서를 입력으로 받음)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# CIFAR-10 데이터셋 로드
# train=True : 학습용 데이터셋 로드
# transform = 데이터셋 로드하면서 정의된 전처리 변환 적용
cifar10_train = datasets.CIFAR10(root="./Data/", train=True, transform=transform, target_transform=None, download=True)


# DataLoader를 통해 배치 단위로 데이터를 로드
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)


# 딥러닝 모델 학습 시 모델, 데이터, parameter 등 모두 동일한 device에 위치해야 됨
# Loss Function : Cross Entroy 사용 (분류 문제)
# Adam optimizer 사용
model = VGG(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_arr = []
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad() #PyTorch는 gradient를 누적하기 때문에 매 loop마다 초기화해야 됨
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward() # Loss에 대한 각 parameter들의 기울기 계산
        optimizer.step() # gradient을 사용해 모델의 parameter를 업데이트

    # detach를 하지 않으면 연산 그래프가 계속 저장 (메모리 낭비)
    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())

torch.save(model.state_dict(), "./models/VGG16_100.pth") 
