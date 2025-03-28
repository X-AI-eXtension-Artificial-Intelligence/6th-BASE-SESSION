import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import trange
from VGG16_SPP import VGG16_SPP


## 데이터셋, 데이터 로더 준비하는 함수
def prepare_data(batch_size=16):
    
    ## 데이터 변환
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    ## CIFAR10 데이터셋
    train_dataset = datasets.CIFAR10(root="./data/", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data/", train=False, transform=transform, download=True)

    ## 데이터 로더 정의
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


## 학습 함수
def train_model(model, train_loader, device, num_epochs=100, lr=0.0002):
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()  ## 손실함수
    optimizer = optim.Adam(model.parameters(), lr=lr)  ## 최적화기법

    loss_arr = []

    for epoch in trange(num_epochs, desc="Epochs"):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            ## 순전파
            outputs = model(images)
            loss = loss_func(outputs, labels)

            ## 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## 5 에포크마다 손실 출력
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        loss_arr.append(loss.item())

    return model, loss_arr


## 모델 저장 함수
def save_model(model, path="vgg16_cifar10.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


## 모델 평가 함수
def evaluate_model(model, test_loader, device):
    correct = 0
    total = 0

    ## 평가모드로 전환
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == '__main__':
    ## 장치 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## 데이터 준비
    batch_size = 64
    train_loader, test_loader = prepare_data(batch_size=batch_size)

    ## 모델 초기화
    model = VGG16_SPP(base_dim=64)

    ## 모델 학습
    num_epochs = 20
    model, loss_arr = train_model(model, train_loader, device, num_epochs=num_epochs, lr=0.0002)

    ## 모델 저장
    save_model(model, path="vgg16_cifar10.pth")

    ## 모델 평가
    evaluate_model(model, test_loader, device)