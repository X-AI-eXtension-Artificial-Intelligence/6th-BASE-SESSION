# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from model import VGG
# #from dataset import data_transform

# import os
# import numpy as np

# # device 설정
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # VGG 클래스를 인스턴스화
# model = VGG(base_dim=64, num_classes=100).to(device)

# batch_size = 100
# learning_rate = 0.0002
# num_epoch = 10

# # 손실함수 및 최적화함수 설정
# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# transform = transforms.Compose([transforms.ToTensor()])
# os.makedirs('./dataset/', exist_ok=True)
# data_set = datasets.CIFAR100(root="./dataset/", train=True, transform=transform, download=True)


# # data

# def data_transform(train=True):
#     # Transform 정의
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     os.makedirs('./dataset/', exist_ok=True)
#     data_set = datasets.CIFAR100(root="./dataset/", train=train, transform=transform, download=True)
#     return data_set


# train_set = data_set
# test_set = data_transform(train=False)

# # DataLoader : 미니배치(batch) 단위로 데이터를 제공
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size)

# loss_arr = []

# for i in range(num_epoch):
#     for [image,label] in train_loader:
#         x = image.to(device)
#         y_ = label.to(device)

#         optimizer.zero_grad()
#         output = model.forward(x)
#         loss = loss_func(output,y_)
#         loss.backward()
#         optimizer.step()

#     if i%10 == 0:
#         print(f'epcoh {i} loss : ', loss)
#         loss_arr.append(loss.cpu().detach().numpy())
# print(np.mean(loss_arr))

# # 학습 완료된 모델 저장
# #torch.save(model.state_dict(), 'VGG.pth')

# print("Training completed. Saving model...")
# torch.save(model.state_dict(), 'VGG.pth')
# print("Model saved as VGG.pth")



import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG

import os
import numpy as np

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터
batch_size = 100
learning_rate = 0.0002
num_epoch = 1000

# 모델 정의
model = VGG(base_dim=64, num_classes=100).to(device)

# 손실 함수 및 옵티마이저
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 데이터셋 및 전처리
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



# 데이터 불러오기
os.makedirs('./dataset/', exist_ok=True)
train_set = datasets.CIFAR100(root="./dataset/", train=True, transform=transform, download=True)
test_set = datasets.CIFAR100(root="./dataset/", train=False, transform=transform, download=True)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# 학습 루프
loss_arr = []

print("Start training...")
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        x = images.to(device)
        y_ = labels.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_arr.append(avg_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

# # 학습 완료 후 저장
# os.makedirs('./checkpoints', exist_ok=True)
# model_path = './checkpoints/VGG.pth'
# torch.save(model.state_dict(), model_path)
# print(f"Model saved at {model_path}")

torch.save(model.state_dict(), 'VGG.pth')
print("✅ Model saved successfully as VGG.pth")
