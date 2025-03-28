import torch  
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# 1x1 컨볼루션
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 1x1 컨볼루션 연산 수행 (입력 채널 수, 출력 채널 수)
        nn.ReLU(),
        nn.BatchNorm2d(out_dim),  # 배치 정규화 추가
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.BatchNorm2d(out_dim),  
        nn.MaxPool2d(2, 2)
    )
    return model 

# 3x3 컨볼루션 
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션 
        nn.ReLU(),
        nn.BatchNorm2d(out_dim),  # 배치 정규화 추가
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(out_dim),  
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.BatchNorm2d(out_dim),  
        nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링으로 맵 크기 감소
    )
    return model  

# VGG16 모델 클래스 
class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=10):  # 클래스 수 입력 받기, 모델 생성
        super(VGG16, self).__init__()  
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),   # 64
            conv_2_block(base_dim, 2 * base_dim),  # 128
            conv_3_block(2 * base_dim, 4 * base_dim),  # 256
            conv_3_block(4 * base_dim, 8 * base_dim),  # 512
            conv_3_block(8 * base_dim, 8 * base_dim),  # 512        
        )
        self.fc_layer = nn.Sequential( 
            nn.Linear(8 * base_dim * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),  # 드롭아웃 비율 증가
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(0.5),  # 드롭아웃 비율 증가
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델, 손실 함수, 옵티마이저 설정
model = VGG16(10).to(device)
loss_func = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Transform 정의 (데이터 전처리)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 데이터셋 로드
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, download=True)
trainloader = DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=2)

cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, download=True)
testloader = DataLoader(cifar10_test, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 이미지 출력 함수
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 무작위 이미지 출력
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
batch_size = images.shape[0]
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# 모델 학습
num_epoch = 30
loss_arr = []
for i in trange(num_epoch):
    model.train()
    epoch_loss = 0

    for j, (image, label) in enumerate(trainloader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    loss_arr.append(epoch_loss / len(trainloader))

    if i % 10 == 0:
        print(f"Epoch [{i}/{num_epoch}], Loss: {epoch_loss / len(trainloader):.4f}")

# 학습 결과를 그래프로 나타내기
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 모델 평가
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for image, label in testloader:
        x = image.to(device)
        y = label.to(device)

        output = model(x)
        _, output_index = torch.max(output, 1)
        total += label.size(0)
        correct += (output_index == y).sum().item()

#정확도
accuracy = 100 * correct / total
print(f"Accuracy of Test Data: {accuracy:.2f}%")
