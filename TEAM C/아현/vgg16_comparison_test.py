import torch
import torchvision.transforms as transforms

from vgg16 import VGG16 as VGG16_Original
from vgg16_light import VGG16 as VGG16_Light
from torch.utils.data import DataLoader
from torchvision import datasets

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

# 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 테스트셋 로딩
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, download=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# 두 모델 각각 로딩
def load_model(model_class, weight_path):
    model = model_class(base_dim=64).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

model_orig = load_model(VGG16_Original, "./train_model")
model_light = load_model(VGG16_Light, "./train_model_light")

# 정확도 계산 함수
def evaluate(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.to(device)
            y = labels.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return 100 * correct / total

# 평가
acc_orig = evaluate(model_orig)
acc_light = evaluate(model_light)

print("======================")
print(f"Original VGG Accuracy: {acc_orig:.2f}%")
print(f"Light VGG Accuracy:    {acc_light:.2f}%")
print("======================")
