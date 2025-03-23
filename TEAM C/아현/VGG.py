import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# conv_2_block 정의
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

# conv_3_block 정의
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

# VGG16 모델 클래스 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        
        # Convolutional layers
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),           # 64
            conv_2_block(base_dim, 2 * base_dim),  # 128
            conv_3_block(2 * base_dim, 4 * base_dim),  # 256
            conv_3_block(4 * base_dim, 8 * base_dim),  # 512
            conv_3_block(8 * base_dim, 8 * base_dim)   # 512        
        )
        
        # Fully connected layers
        self.fc_layer = nn.Sequential(
            # CIFAR10은 크기가 32x32이므로 
            nn.Linear(8 * base_dim * 1 * 1, 4096),  # Flatten 후 입력
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layer(x)
        return x

# 모델 학습을 위한 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 모델 인스턴스화
model = VGG(base_dim=64).to(device)

# 손실 함수와 최적화 함수 설정
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# CIFAR10 데이터셋 로드
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 학습 데이터셋과 테스트 데이터셋 정의
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, download=True)
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, download=True)

# DataLoader 설정
train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=64, shuffle=False)

# 학습용 이미지를 무작위로 가져오기 및 시각화
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습 데이터에서 이미지를 하나 가져오기
dataiter = iter(train_loader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))

# 클래스 이름 출력
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(' '.join('%5s' % classes[labels[j]] for j in range(64)))
