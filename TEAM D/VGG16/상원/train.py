import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16_Improved import VGG16_Improved  # 개선된 VGG 모델 사용

# ---------------- 설정 ----------------
# 학습률 0.002
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ---------------- 전처리 ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- 데이터 로드 ----------------
train_dataset = datasets.CIFAR10(root='./Data/', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------- 모델, 손실, 옵티마이저 ----------------
model = VGG16_Improved(base_dim=64, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---------------- 학습 ----------------
loss_arr = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_arr.append(avg_loss)

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

# ---------------- 모델 저장 ----------------
torch.save(model.state_dict(), "./trained_vgg16_improved.pth")
