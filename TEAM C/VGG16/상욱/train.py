import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from model import VGG
from model_update import VGG
import torch.nn as nn
import torch

batch_size = 100
learning_rate = 0.0002
num_epoch = 10

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Transform 정의
# 이미지 데이터를 텐서로 변환
# transforms.Normalize(mean, std)
# channel = 3
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # 랜덤 크롭 후 리사이즈
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    transforms.RandomRotation(degrees=10),  # 최대 ±10도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
    transforms.ToTensor(),  # 텐서 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 정규화
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 일부 픽셀 제거 (Cutout)
])

# CIFAR10 TRAIN 데이터 정의
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, target_transform=None, download=True)


# train_loader 정의
train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers = 2) # num_workers를 통해 더 빨리 할 수 있음

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# VGG 클래스를 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실함수 및 최적화함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []
for i in range(num_epoch):
    # train_loader에서 이미지를 하나씩 가져와서 처리
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad() # 기울기(gradient)를 0으로 초기화
        output = model.forward(x) # 모델에 이미지를 넣어 예측값을 생성
        loss = loss_func(output,y_) # 예측값과 실제 라벨의 차이를 계산 (손실값 계산)
        loss.backward() # 손실값을 바탕으로 기울기 계산 (역전파)
        optimizer.step() # 계산된 기울기를 바탕으로 모델 파라미터 업데이트

    if i % 10 ==0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())

# 훈련이 끝난 모델 save
torch.save(model.state_dict(), 'vgg_model_update.pth')
print("Model saved as 'vgg_model_update.pth'")
