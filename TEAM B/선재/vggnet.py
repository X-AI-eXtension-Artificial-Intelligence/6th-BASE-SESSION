import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange  # 진행률 표시용

# Hyperparameters
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 데이터셋 다운로드 및 DataLoader 설정
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CNN 블록 정의
def conv_2_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

def conv_3_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

# VGG 모델 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 64
            conv_2_block(base_dim, 2*base_dim),  # 128
            conv_3_block(2*base_dim, 4*base_dim),  # 256
            conv_3_block(4*base_dim, 8*base_dim),  # 512
            conv_3_block(8*base_dim, 8*base_dim),  # 512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim, 4096),  # 크기 문제 수정
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성 및 GPU 이동
model = VGG(base_dim=64).to(device)

# 손실 함수 및 최적화 함수 설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 이미지 출력 함수
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져와서 출력
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))

# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

# 학습 시작
loss_arr = []
for i in trange(num_epoch):
    model.train()
    epoch_loss = 0  # 한 epoch당 평균 loss 계산용

    for j, (image, label) in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # loss 누적

    loss_arr.append(epoch_loss / len(train_loader))  # 평균 loss 저장
    if i % 10 == 0:
        print(f"Epoch [{i}/{num_epoch}], Loss: {epoch_loss / len(train_loader):.4f}")

# 학습 결과 그래프 출력
plt.plot(loss_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 모델 평가
correct = 0
total = 0

model.eval()  # 평가 모드

# 인퍼런스 모드를 위해 no_grad 사용
with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y = label.to(device)

        output = model(x)
        _, output_index = torch.max(output, 1)

        total += label.size(0)
        correct += (output_index == y).sum().item()  # `.item()` 사용하여 CPU에서 계산

# 정확도 출력
accuracy = 100 * correct / total
print(f"Accuracy of Test Data: {accuracy:.2f}%")
